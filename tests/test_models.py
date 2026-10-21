"""Model/loss/eval tests on CPU with synthetic tensors (no experimental numbers)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import pad_bags  # noqa: E402
from src.evaluation.calibration import fit_temperature  # noqa: E402
from src.evaluation.metrics import ece, task_metrics  # noqa: E402
from src.evaluation.statistics import bootstrap_ci, mcnemar, paired_bootstrap  # noqa: E402
from src.models.baseline import ImageLevelModel  # noqa: E402
from src.models.core_mil import GatedAttention, MILModel  # noqa: E402
from src.training.losses import class_weights, cross_view_consistency, supervised_loss, warmup  # noqa: E402

NC = {"hp": 2, "atrophy": 2, "distribution": 4, "type": 3}
M = {"atrophy": np.array([[0, 1], [1, 0], [1, 0], [1, 0]], np.float32),
     "type": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]], np.float32)}


def bags(D=16, sizes=(3, 7, 1, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(sum(sizes), D, generator=g)
    rows, s = [], 0
    for n in sizes:
        rows.append(np.arange(s, s + n))
        s += n
    return feats, rows


# 6. variable-sized bags -----------------------------------------------------
def test_pad_bags_shapes_and_zero_padding():
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    assert x.shape == (4, 7, 16) and mask.sum(1).tolist() == [3, 7, 1, 5]
    assert (x[~mask] == 0).all()
    assert torch.equal(x[1, :7], feats[3:10])


# 7. attention masks ---------------------------------------------------------
def test_attention_ignores_padding_and_is_invariant_to_it():
    torch.manual_seed(0)
    att = GatedAttention(16, 8, n_branches=3)
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    a = att(x, mask)
    assert a.shape == (4, 3, 7)
    assert torch.allclose(a.sum(-1), torch.ones(4, 3))
    assert (a.permute(0, 2, 1)[~mask] == 0).all()
    x2 = x.clone()
    x2[~mask] = 1e3  # garbage in padded slots must not change anything
    assert torch.allclose(att(x2, mask), a)
    # different task branches attend differently
    assert not torch.allclose(a[:, 0], a[:, 1])


# 8. forward/backward passes ------------------------------------------------
@pytest.mark.parametrize("pooling", ["mean", "attention", "task_attention"])
@pytest.mark.parametrize("head", ["independent", "structured"])
def test_mil_forward_backward(pooling, head):
    torch.manual_seed(0)
    model = MILModel(16, NC, pooling=pooling, head=head, mapping=M, hidden=32, attn_dim=8)
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    out = model(x, mask)
    lp = model.log_probs(out)
    assert set(lp) == set(NC)
    for t, c in NC.items():
        assert lp[t].shape == (4, c)
        assert torch.allclose(lp[t].exp().sum(1), torch.ones(4), atol=1e-5)
    y = torch.tensor([[0, 1, 0, 2], [1, 0, 1, 1], [0, 0, 2, 0], [1, 0, 3, 1]])
    loss, per = supervised_loss(lp, y)
    loss.backward()
    assert all(p.grad is not None for p in model.heads.parameters())
    if head == "structured":
        # dependent tasks are exact functions of the distribution distribution
        pdist = lp["distribution"].exp()
        assert torch.allclose(lp["type"].exp(), pdist @ torch.as_tensor(M["type"]), atol=1e-5)


def test_mil_padding_invariance_of_predictions():
    torch.manual_seed(0)
    model = MILModel(16, NC, pooling="task_attention", head="structured", mapping=M, hidden=32).eval()
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    single = model(x[2:3, :1], mask[2:3, :1])["logits"]["distribution"]
    batched = model(x, mask)["logits"]["distribution"][2:3]
    assert torch.allclose(single, batched, atol=1e-5)


def test_image_level_aggregation_is_mean_probability():
    model = ImageLevelModel(16, NC, hidden=8).eval()
    feats, _ = bags()
    pos = torch.tensor([0] * 3 + [1] * 13)
    z = model(feats)
    agg = ImageLevelModel.aggregate(z, pos, 2)
    assert torch.allclose(agg["distribution"][0].exp(), z["distribution"][:3].softmax(-1).mean(0), atol=1e-6)


def test_missing_labels_are_masked():
    lp = {t: torch.log_softmax(torch.randn(3, c), -1) for t, c in NC.items()}
    y = torch.tensor([[0, -1, 1, 0], [1, -1, 2, 1], [0, -1, 3, 2]])
    _, per = supervised_loss(lp, y)
    assert "atrophy" not in per and set(per) == {"hp", "distribution", "type"}


def test_class_weights_and_focal():
    y = np.array([[0, 0, 0, 0]] * 9 + [[1, 1, 1, 1]])
    w = class_weights(y, NC)
    assert w["hp"][1] > w["hp"][0] and w["distribution"][3] == 0
    lp = {t: torch.log_softmax(torch.randn(10, c), -1) for t, c in NC.items()}
    ce, _ = supervised_loss(lp, torch.as_tensor(y), "ce")
    fl, _ = supervised_loss(lp, torch.as_tensor(y), "focal", gamma=2.0)
    assert fl < ce


# 9. cross-view consistency loss --------------------------------------------
def test_consistency_gating_detach_and_zero_at_agreement():
    zt = {"distribution": torch.tensor([[5.0, 0, 0, 0], [0.1, 0, 0, 0]], requires_grad=True)}
    zs = {"distribution": torch.zeros(2, 4, requires_grad=True)}
    loss, gate = cross_view_consistency(zt, zs, temperature=2.0, confidence=0.7)
    assert gate == 0.5  # only the confident teacher passes
    loss.backward()
    assert zt["distribution"].grad is None  # teacher detached
    assert zs["distribution"].grad[1].abs().sum() == 0  # gated-out patient gets no gradient
    assert zs["distribution"].grad[0].abs().sum() > 0
    same = {"distribution": zt["distribution"].detach().clone()}
    l0, _ = cross_view_consistency(same, {"distribution": same["distribution"].clone()}, 2.0, 0.7)
    assert l0.abs() < 1e-6
    assert warmup(2, 5, 10) == 0 and warmup(5, 5, 10) == 0.1 and warmup(40, 5, 10) == 1.0


# 10. calibration ------------------------------------------------------------
def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, C = 4000, 4
    true_logits = rng.normal(size=(n, C)) * 1.5
    p = np.exp(true_logits) / np.exp(true_logits).sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=pi) for pi in p])
    T = fit_temperature(true_logits * 3.0, y)  # model 3x overconfident
    assert 2.5 < T < 3.5
    sharp = np.exp(true_logits * 3)
    sharp /= sharp.sum(1, keepdims=True)
    assert ece(y, p) < ece(y, sharp)


# 12. metrics and output generation ------------------------------------------
def test_task_metrics_values_and_undefined_auc():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.eye(3)[[0, 0, 1, 2, 2, 2]] * 0.8 + 0.2 / 3
    m = task_metrics(y, P, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["confusion"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert m["recall[b]"] == 0.5 and 0 <= m["ece"] <= 1 and m["auroc"] > 0.5
    m2 = task_metrics(np.array([0, 0, 1]), np.eye(3)[[0, 0, 1]] * 0.9 + 0.1 / 3, ["a", "b", "c"])
    assert np.isnan(m2["auroc"]) and m2["classes_absent"] == ["c"] and np.isnan(m2["recall[c]"])


def test_statistics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 60)
    good = np.eye(3)[y] * 0.7 + 0.1
    bad = np.full((60, 3), 1 / 3) + rng.normal(0, 0.01, (60, 3))
    ci = bootstrap_ci(y, good, n_boot=200)
    assert ci["lo"] <= ci["estimate"] <= ci["hi"] and ci["estimate"] == 1.0
    d = paired_bootstrap(y, good, bad, n_boot=200)
    assert d["diff"] > 0 and d["lo"] > 0 and d["p_boot"] < 0.05
    mc = mcnemar(y, good.argmax(1), bad.argmax(1))
    assert mc["a_only_correct"] > mc["b_only_correct"] and mc["p_exact"] < 0.05


def test_fast_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from src.evaluation.aggregate import _fast_metric
    rng = np.random.default_rng(1)
    for C in (2, 3, 4):
        y = rng.integers(0, C - 1, 50)  # last class absent from y on purpose
        P = rng.random((50, C))
        pred = P.argmax(1)
        assert _fast_metric(y, P, "macro_f1") == pytest.approx(
            f1_score(y, pred, labels=np.arange(C), average="macro", zero_division=0))
        assert _fast_metric(y, P, "accuracy") == pytest.approx(accuracy_score(y, pred))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _fast_metric(y, P, "balanced_accuracy") == pytest.approx(balanced_accuracy_score(y, pred))


# Addendum B aggregators --------------------------------------------------------
def test_topk_mean_matches_manual_and_handles_short_bags():
    from src.models.core_mil import topk_mean
    torch.manual_seed(0)
    feats, rows = bags(D=3, sizes=(9, 2, 5))
    x, mask = pad_bags(feats, rows)
    out = topk_mean(x, mask, k=3)
    for b, r in enumerate(rows):
        v = feats[r]
        kk = min(3, len(r))
        assert torch.allclose(out[b], v.topk(kk, dim=0).values.mean(0), atol=1e-6)


@pytest.mark.parametrize("pooling", ["topk", "transformer"])
@pytest.mark.parametrize("head", ["independent", "structured"])
def test_new_poolings_forward_backward_and_padding_invariance(pooling, head):
    torch.manual_seed(0)
    model = MILModel(16, NC, pooling=pooling, head=head, mapping=M, hidden=32, tr_heads=4, tr_ffn=64)
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    lp = model.log_probs(model(x, mask))
    loss, _ = supervised_loss(lp, torch.tensor([[0, 1, 0, 2], [1, 0, 1, 1], [0, 0, 2, 0], [1, 0, 3, 1]]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    x2 = x.clone()
    x2[~mask] = 1e3  # padded content must not matter
    with torch.no_grad():
        a = model(x, mask)["logits"]["distribution"]
        b = model(x2, mask)["logits"]["distribution"]
        single = model(x[2:3, :1], mask[2:3, :1])["logits"]["distribution"]
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a[2:3], single, atol=1e-5)


def test_acmil_masking_only_in_training_and_keeps_one_image():
    torch.manual_seed(0)
    att = GatedAttention(16, 8, n_branches=2, mask_topk=5, mask_p=1.0)  # p=1: mask every allowed top image
    plain = GatedAttention(16, 8, n_branches=2)
    plain.load_state_dict(att.state_dict())
    feats, rows = bags(sizes=(3, 7, 1, 5))
    x, mask = pad_bags(feats, rows)
    att.eval(); plain.eval()
    assert torch.allclose(att(x, mask), plain(x, mask))  # inference is unmasked
    att.train()
    a = att(x, mask)
    assert torch.allclose(a.sum(-1), torch.ones(4, 2))  # still a distribution
    nz = (a > 0).sum(-1)  # images with non-zero weight per bag/branch
    expect = torch.tensor([max(1, n - min(5, n - 1)) for n in (3, 7, 1, 5)])
    assert torch.equal(nz, expect.view(4, 1).expand(4, 2))


def test_instance_logits_share_patient_heads():
    """Addendum C: image-level logits use the patient heads, so attention pooling of them gives the patient logit."""
    torch.manual_seed(0)
    model = MILModel(16, NC, pooling="attention", head="structured", mapping=M, hidden=8, attn_dim=4).eval()
    feats, rows = bags()
    x, mask = pad_bags(feats, rows)
    out = model(x, mask)
    inst = model.instance_logits(out["h"])  # [B, K, C]
    for g in model.groups:
        pooled = (out["attn"][:, 0].unsqueeze(-1) * inst[g]).sum(1)
        assert torch.allclose(pooled, out["logits"][g], atol=1e-5)
    lp = model.log_probs(dict(logits=model.instance_logits(out["h"][mask])))
    assert lp["type"].shape == (int(mask.sum()), 3)  # derived tasks are available per image
    y = torch.zeros(int(mask.sum()), 4, dtype=torch.long)
    supervised_loss(lp, y)[0].backward()
    assert model.embed[2].weight.grad.abs().sum() > 0
