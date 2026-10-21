"""Data-side tests: metadata parsing, grouping, duplicates, splits, mappings, view sampling.

Tests marked `real` read data/ (read-only) and are skipped if it is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audit import label_relationships, patient_groups  # noqa: E402
from src.data.dataset import sample_view  # noqa: E402
from src.data.metadata import TASK_KEYS, _tokens, discover_tasks, label_table, load_metadata  # noqa: E402
from src.data.splits import PAPER_TEST, allocate, check_split, cv_splits, primary_split, strata_table  # noqa: E402
from src.evaluation.robustness import eval_subset  # noqa: E402
from src.models.structured_head import coherence, derive_mapping  # noqa: E402

DATA = ROOT / "data"
real = pytest.mark.skipif(not (DATA / "metadata.xlsx").exists(), reason="EGID data not present")


def synthetic_meta(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dist = rng.choice(["normal", "antrum_predominant", "corpus_predominant", "antrum_and_corpus"], n)
    hp = np.where(dist == "corpus_predominant", "negative", rng.choice(["positive", "negative"], n))
    df = pd.DataFrame(dict(Patient_ID=[f"{i:03d}" for i in range(1, n + 1)], Number_of_pictures=10,
                           H_pylori_infection_status=hp, Distribution_of_atrophy=dist,
                           Presence_of_atrophy=np.where(dist == "normal", "non_atrophic", "atrophic"),
                           Gastritis_type=pd.Series(dist).map({"normal": "normal", "corpus_predominant": "type_A",
                                                               "antrum_predominant": "type_B",
                                                               "antrum_and_corpus": "type_B"}).values))
    return df.set_index("Patient_ID", drop=False)


# 1. metadata parsing --------------------------------------------------------
def test_token_matching_handles_dataset_spellings():
    assert _tokens("4.corpus_and_antrum") == _tokens("antrum_and_corpus")
    assert _tokens("2.non-atrophic") == _tokens("non_atrophic")
    assert _tokens("1.type_A") != _tokens("type_B")


@real
def test_real_metadata_and_folders_agree():
    meta = load_metadata(DATA)
    assert len(meta) == 229 and meta.Patient_ID.str.len().eq(3).all()
    tasks = discover_tasks(DATA, meta)
    assert [t.key for t in tasks] == TASK_KEYS
    assert [t.n_classes for t in tasks] == [2, 2, 4, 3]
    lab = label_table(meta, tasks)
    assert (lab.to_numpy() >= 0).all()


# 2. patient grouping & 3. duplicate linkage --------------------------------
def test_patient_groups_union_find():
    pairs = pd.DataFrame(dict(patient_a=["002", "003"], patient_b=["005", "002"]))
    g = patient_groups(["001", "002", "003", "005"], pairs)
    assert g["002"] == g["003"] == g["005"] != g["001"]


@real
def test_audit_outputs_have_no_cross_patient_duplicates():
    audit = ROOT / "outputs/data_audit"
    if not (audit / "image_index.csv").exists():
        pytest.skip("run scripts/audit_data.py first")
    idx = pd.read_csv(audit / "image_index.csv", dtype={"patient_id": str})
    assert idx.image_id.is_unique and len(idx) == 5883
    by_hash = idx.groupby("sha256").patient_id.nunique()
    assert (by_hash == 1).all(), "identical file bytes shared by different patients"


# 4. patient-level splitting -------------------------------------------------
def test_primary_split_matches_paper_composition_and_is_disjoint():
    meta = synthetic_meta(229, 1)
    groups = {p: p for p in meta.index}
    groups["002"] = groups["007"] = "002"  # a duplicate-linked pair must stay together
    st = strata_table(meta, groups)
    sp = primary_split(st, seed=0)
    check_split(sp, groups)
    assert len(sp["test"]) == 34 and len(set(sp["train"]) | set(sp["val"]) | set(sp["test"])) == 229
    where = {p: k for k in ("train", "val", "test") for p in sp[k]}
    assert where["002"] == where["007"]


def test_allocate_hits_exact_marginals_when_feasible():
    sizes = {"negative|antrum_and_corpus": 31, "positive|antrum_and_corpus": 29, "negative|antrum_predominant": 31,
             "positive|antrum_predominant": 43, "negative|corpus_predominant": 31, "negative|normal": 32,
             "positive|normal": 32}
    a = allocate(sizes, 34, PAPER_TEST)
    assert sum(a.values()) == 34
    assert a["negative|corpus_predominant"] == 5
    assert sum(v for k, v in a.items() if k.startswith("negative")) == 19


def test_cv_folds_partition_development_cohort():
    meta = synthetic_meta(150, 2)
    groups = {p: p for p in meta.index}
    st = strata_table(meta, groups)
    folds = cv_splits(st, list(meta.index[:130]), k=5, val_frac=0.15, seed=0)
    tests = [set(f["test"]) for f in folds]
    assert set().union(*tests) == set(meta.index[:130])
    assert sum(len(t) for t in tests) == 130
    for f in folds:
        check_split(f, groups)
        assert not set(f["train"] + f["val"] + f["test"]) & set(meta.index[130:])


# 5. clinical label mappings -------------------------------------------------
def test_mapping_derived_and_rejected_correctly():
    meta = synthetic_meta(200, 3)
    from src.data.metadata import Task
    tasks = [Task("hp", "", "H_pylori_infection_status", ["positive", "negative"]),
             Task("atrophy", "", "Presence_of_atrophy", ["atrophic", "non_atrophic"]),
             Task("distribution", "", "Distribution_of_atrophy",
                  ["normal", "antrum_predominant", "corpus_predominant", "antrum_and_corpus"]),
             Task("type", "", "Gastritis_type", ["type_A", "type_B", "normal"])]
    y = label_table(meta, tasks).to_numpy()
    nc = {t.key: t.n_classes for t in tasks}
    M = derive_mapping(y, nc)
    assert M is not None
    assert M["type"][2].argmax() == 0 and M["type"][0].argmax() == 2  # corpus -> type_A, normal -> normal
    assert M["atrophy"][0].argmax() == 1  # normal -> non_atrophic
    pred = {"distribution": y[:, 2], "atrophy": y[:, 1], "type": y[:, 3]}
    assert coherence(pred, M).all()
    y_bad = y.copy()
    y_bad[0, 3] = (y_bad[0, 3] + 1) % 3  # one inconsistent patient breaks determinism
    assert derive_mapping(y_bad, nc) is None


def test_label_relationship_audit_flags_determinism():
    _, findings, _ = label_relationships(synthetic_meta(100, 4))
    f = {(x["source"], x["target"]): x["deterministic"] for x in findings}
    assert f[("Distribution_of_atrophy", "Gastritis_type")]
    assert not f[("Distribution_of_atrophy", "H_pylori_infection_status")]
    assert not f[("H_pylori_infection_status", "Distribution_of_atrophy")]


# 11. missing-view sampling --------------------------------------------------
@pytest.mark.parametrize("mode", ["subset", "bernoulli", "fixed_k"])
def test_view_sampling_never_empty_and_within_bag(mode):
    vd = dict(mode=mode, teacher_ratio=[1.0, 1.0], student_ratio=[0.0, 0.3], bernoulli_p=0.99, fixed_k=[2, 12])
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 50):
        for which in ("teacher", "student"):
            v = sample_view(n, rng, vd, which)
            assert 1 <= len(v) <= n and len(set(v)) == len(v) and v.max() < n
    assert len(sample_view(30, rng, vd, "teacher")) == 30 if mode != "fixed_k" else True


def test_eval_subsets_reproducible_and_model_independent():
    a = eval_subset("017", 40, 0.5, rep=3)
    assert np.array_equal(a, eval_subset("017", 40, 0.5, rep=3))
    assert len(a) == 20 and not np.array_equal(a, eval_subset("017", 40, 0.5, rep=4))
    assert len(eval_subset("017", 3, 0.25, 0)) == 1  # at least one image
    assert np.array_equal(eval_subset("017", 9, 1.0, 0), np.arange(9))
