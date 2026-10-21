"""Cached-feature training/evaluation for image-level baselines and MIL variants.

One call of `run` = one (variant, encoder, split, seed). It trains with early
stopping on validation patients, fits temperature on validation patients,
evaluates on the split's held-out patients at several image-availability
fractions, and writes a machine-readable record. Existing records are never
overwritten (re-running skips them).
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import environment, seed_everything
from ..data.dataset import Cohort, pad_bags, sample_view
from ..data.metadata import TASK_KEYS
from ..data.splits import load_split
from ..evaluation.calibration import fit_temperature
from ..evaluation.metrics import all_task_metrics
from ..evaluation.robustness import eval_subset
from ..models.baseline import ImageLevelModel
from ..models.core_mil import MILModel
from ..models.structured_head import derive_mapping, finalize, soft_clinical_loss
from .losses import class_weights, cross_view_consistency, supervised_loss, warmup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    variant: str
    encoder: str
    split: str
    seed: int
    size: int = 224

    @property
    def run_id(self) -> str:
        return f"{self.variant}__{self.encoder}_{self.size}__{self.split}__s{self.seed}"


def variant_config(cfg: dict, variant: str, profile: dict) -> dict:
    v = dict(cfg["train"])
    var = cfg["variants"][variant]
    # tuned hyper-parameters (selected on inner-validation NLL by scripts/tune.py) per model kind;
    # explicit variant keys win over them, profile overrides win over everything
    if not var.get("finetune"):
        v.update(cfg.get("tuned", {}).get(var["kind"], {}))
    v.update(var)
    v.update(profile.get("overrides", {}))
    for k in ("max_epochs", "patience"):
        if k in profile:
            v[k] = profile[k]
    v.setdefault("view_dropout", False)
    v.setdefault("consistency", False)
    v.setdefault("clinical", False)
    if v["consistency"] and not v["view_dropout"]:
        raise ValueError(f"{variant}: cross-view consistency needs two views (set view_dropout: true)")
    return v


class Runner:
    def __init__(self, cfg: dict, co: Cohort, feats: np.ndarray, spec: RunSpec, device: torch.device,
                 profile: dict, out_root: Path):
        self.cfg, self.co, self.spec, self.device, self.profile = cfg, co, spec, device, profile
        self.v = variant_config(cfg, spec.variant, profile)
        self.out = out_root
        sp = load_split(cfg["paths"]["splits"], spec.split)
        self.split = sp
        train = sp["train"]
        if profile.get("max_train_patients"):
            rng = np.random.default_rng(spec.seed)
            train = sorted(rng.choice(train, size=min(len(train), profile["max_train_patients"]), replace=False))
        self.ids = dict(train=list(train), val=sp["val"], test=sp["test"])
        self.y_train = co.y(self.ids["train"])
        self.mapping_np = derive_mapping(self.y_train, co.n_classes)
        head = self.v.get("head", "independent")
        if head == "structured" and self.mapping_np is None:
            log.warning("%s: falling back to independent heads (mapping not deterministic on training data)", spec.run_id)
            head = "independent"
        self.head = head
        self.mapping = {k: torch.as_tensor(m, device=device) for k, m in self.mapping_np.items()} \
            if self.mapping_np else None
        # z-score features with training-image statistics only
        rows, _ = co.image_rows(self.ids["train"])
        f = torch.as_tensor(feats, dtype=torch.float32)
        mu, sd = (f[rows].mean(0), f[rows].std(0).clamp_min(1e-6)) if self.v["feature_standardize"] else (0.0, 1.0)
        self.X = ((f - mu) / sd).to(device)
        # Addendum C: augmented feature versions for training only (clean statistics; version 0 = clean)
        self.X_views = None
        if self.v.get("aug_views"):
            from .feature_extraction import load_aug_features
            aug = torch.as_tensor(load_aug_features(cfg["paths"]["cache"], spec.encoder, spec.size, self.v["aug_views"],
                                                    co.index.image_id.tolist()))
            self.X_views = torch.cat([self.X.unsqueeze(0).cpu(), (aug - mu) / sd]).to(device)
        seed_everything(spec.seed)
        kw = dict(hidden=self.v["hidden"], dropout=self.v["dropout"], input_dropout=self.v["input_dropout"])
        if self.v["kind"] == "image":
            self.model = ImageLevelModel(feats.shape[1], co.n_classes, **kw)
        else:
            extra = {k: self.v[k] for k in ("topk", "mask_topk", "mask_p", "tr_layers", "tr_heads", "tr_ffn") if k in self.v}
            self.model = MILModel(feats.shape[1], co.n_classes, pooling=self.v["pooling"], head=head,
                                  mapping=self.mapping_np, attn_dim=self.v["attn_dim"], **kw, **extra)
        self.model.to(device)
        self.weights = class_weights(self.y_train, co.n_classes)
        self.rng = np.random.default_rng(spec.seed)

    # ---------------------------------------------------------------- training
    def _sup(self, logp, y):
        return supervised_loss(logp, y, self.v["supervised_loss"], self.weights, self.v["focal_gamma"])

    def _patient_order(self) -> list[str]:
        ids = self.ids["train"]
        if self.v["sampling"] == "balanced":
            d = self.y_train[:, TASK_KEYS.index("distribution")]
            w = 1.0 / np.bincount(d)[d]
            return list(self.rng.choice(ids, size=len(ids), replace=True, p=w / w.sum()))
        return list(self.rng.permutation(ids))

    def _epoch_mil(self, epoch: int) -> dict:
        m, v, co = self.model, self.v, self.co
        vd, cons = self.cfg["view_dropout"], self.cfg["consistency"]
        lam2 = cons["lambda"] * warmup(epoch, cons["warmup_start"], cons["warmup_epochs"]) if v["consistency"] else 0.0
        X = self.X
        if self.X_views is not None:  # one of the clean/augmented versions per image for this epoch
            n = X.shape[0]
            k = torch.as_tensor(self.rng.integers(len(self.X_views), size=n), device=self.device)
            X = self.X_views[k, torch.arange(n, device=self.device)]
        order = self._patient_order()
        logs = []
        for s in range(0, len(order), v["batch_patients"]):
            pids = order[s: s + v["batch_patients"]]
            y = torch.as_tensor(co.y(pids), device=self.device)
            bags = [co.bags[p] for p in pids]
            if v["view_dropout"]:
                rows_t = [b[sample_view(len(b), self.rng, vd, "teacher")] for b in bags]
                rows_s = [b[sample_view(len(b), self.rng, vd, "student")] for b in bags]
            else:
                rows_t, rows_s = bags, None
            x_t, mask_t = pad_bags(X, rows_t)
            out_t = m(x_t, mask_t)
            lp_t = m.log_probs(out_t)
            loss, _ = self._sup(lp_t, y)
            rec = dict(sup=loss.item(), cv=0.0, clin=0.0, gate=np.nan, img=0.0)
            if rows_s is not None:
                out_s = m(*pad_bags(X, rows_s))
                l_s, _ = self._sup(m.log_probs(out_s), y)
                loss = (loss + l_s) / 2
                rec["sup"] = loss.item()
                if v["consistency"]:
                    l_cv, gate = cross_view_consistency(out_t["logits"], out_s["logits"], cons["temperature"],
                                                        cons["confidence"])
                    rec.update(cv=l_cv.item(), gate=gate)
                    loss = loss + lam2 * l_cv
            if v.get("img_aux"):  # Addendum C: every image of the teacher view carries its patient's labels
                lp_i = m.log_probs(dict(logits=m.instance_logits(out_t["h"][mask_t])))
                l_img, _ = self._sup(lp_i, y.repeat_interleave(mask_t.sum(1), dim=0))
                rec["img"] = l_img.item()
                loss = loss + v["img_aux"] * l_img
            if v["clinical"] and self.head == "independent" and self.mapping:
                l_c = soft_clinical_loss(lp_t, self.mapping)
                rec["clin"] = l_c.item()
                loss = loss + self.cfg["clinical"]["lambda"] * l_c
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            logs.append(rec)
        out = pd.DataFrame(logs).mean().to_dict()
        out["lambda2"] = lam2
        return out

    def _epoch_image(self, epoch: int) -> dict:
        m, v, co = self.model, self.v, self.co
        rows, pos = co.image_rows(self.ids["train"])
        y_img = self.y_train[pos]
        perm = self.rng.permutation(len(rows))
        logs = []
        for s in range(0, len(perm), v["batch_images"]):
            b = perm[s: s + v["batch_images"]]
            lp = finalize(m(self.X[torch.as_tensor(rows[b], device=self.device)]), None)
            loss, _ = self._sup(lp, torch.as_tensor(y_img[b], device=self.device))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            logs.append(dict(sup=loss.item()))
        return dict(sup=float(np.mean([r["sup"] for r in logs])), cv=0.0, clin=0.0, gate=np.nan, lambda2=0.0)

    # -------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict(self, pids: list[str], frac: float = 1.0, rep: int = 0, want_attn: bool = False) -> dict:
        """Patient-level per-group logits (MIL) or log mean probabilities (image-level)."""
        m, co = self.model.eval(), self.co
        rows = [co.bags[p][eval_subset(p, len(co.bags[p]), frac, rep)] for p in pids]
        if self.v["kind"] == "image":
            flat = np.concatenate(rows)
            pos = torch.as_tensor(np.repeat(np.arange(len(pids)), [len(r) for r in rows]), device=self.device)
            base = ImageLevelModel.aggregate(m(self.X[torch.as_tensor(flat, device=self.device)]), pos, len(pids))
            res = dict(logits={g: z.cpu().numpy() for g, z in base.items()}, rows=rows)
        else:
            outs, attn = [], []
            for s in range(0, len(pids), 32):
                o = m(*pad_bags(self.X, rows[s: s + 32]))
                outs.append({g: z.cpu().numpy() for g, z in o["logits"].items()})
                if want_attn and o["attn"] is not None:
                    attn += [a[:, : len(r)].cpu().numpy() for a, r in zip(o["attn"], rows[s: s + 32])]
            res = dict(logits={g: np.concatenate([o[g] for o in outs]) for g in outs[0]}, rows=rows, attn=attn)
        self.model.train()
        return res

    def probs(self, logits: dict[str, np.ndarray], temps: dict | None = None) -> dict[str, np.ndarray]:
        t = {g: torch.as_tensor(z) for g, z in logits.items()}
        mp = {k: v.cpu() for k, v in self.mapping.items()} if (self.mapping and self.head == "structured") else None
        return {k: v.exp().numpy() for k, v in finalize(t, mp, temps).items()}

    def val_score(self) -> tuple[float, dict]:
        P = self.probs(self.predict(self.ids["val"])["logits"])
        y = self.co.y(self.ids["val"])
        nll = float(np.mean([-np.log(np.clip(P[t][np.arange(len(y)), y[:, j]], 1e-12, None)).mean()
                             for j, t in enumerate(TASK_KEYS)]))
        from sklearn.metrics import f1_score
        j = TASK_KEYS.index("distribution")
        f1 = f1_score(y[:, j], P["distribution"].argmax(1), average="macro", zero_division=0)
        return nll, dict(val_nll=nll, val_dist_f1=f1)

    # --------------------------------------------------------------------- run
    def fit(self) -> dict:
        v = self.v
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=v["lr"], weight_decay=v["weight_decay"])
        best, best_state, best_epoch, bad, hist = np.inf, None, -1, 0, []
        t0 = time.time()
        for epoch in range(v["max_epochs"]):
            tr = self._epoch_image(epoch) if v["kind"] == "image" else self._epoch_mil(epoch)
            score, vl = self.val_score()
            hist.append(dict(epoch=epoch, **tr, **vl))
            if score < best - 1e-4:
                best, best_state, best_epoch, bad = score, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                bad += 1
                # never stop while the consistency term is still ramping in
                ramping = v["consistency"] and epoch < self.cfg["consistency"]["warmup_start"] + \
                    self.cfg["consistency"]["warmup_epochs"]
                if bad >= v["patience"] and not ramping:
                    break
        self.model.load_state_dict(best_state)
        return dict(best_epoch=best_epoch, epochs_run=len(hist), best_val_nll=best, train_seconds=time.time() - t0,
                    history=hist)

    def evaluate(self, fracs: list[float], reps: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        co = self.co
        val = self.predict(self.ids["val"])
        yv = co.y(self.ids["val"])
        temps = {g: fit_temperature(z, yv[:, TASK_KEYS.index(g)]) for g, z in val["logits"].items()}
        res, pred_rows = dict(temperatures=temps), []
        mp = self.mapping_np
        for name, pids in (("val", self.ids["val"]), ("test", self.ids["test"])):
            y = co.y(pids)
            for frac in (fracs if name == "test" else [1.0]):
                for rep in range(1 if frac >= 1 else reps):
                    pr = val if (name == "val") else self.predict(pids, frac, rep, want_attn=(frac == 1.0))
                    if name == "test" and frac == 1.0:
                        test_full = pr
                    for cal, T in (("raw", None), ("cal", temps)):
                        P = self.probs(pr["logits"], T)
                        res[f"{name}|{frac}|{rep}|{cal}"] = all_task_metrics(y, P, co.tasks, mp)
                        for j, t in enumerate(TASK_KEYS):
                            for i, p in enumerate(pids):
                                pred_rows.append(dict(set=name, frac=frac, rep=rep, cal=cal, patient_id=p, task=t,
                                                      label=int(y[i, j]), n_images=len(pr["rows"][i]),
                                                      **{f"p{c}": float(P[t][i, c]) for c in range(P[t].shape[1])}))
        attn_rows = []
        for i, (p, a) in enumerate(zip(self.ids["test"], test_full.get("attn", []))):
            ids = co.index.image_id.to_numpy()[test_full["rows"][i]]
            br = self.model.groups if a.shape[0] > 1 else ["shared"]
            for b, g in enumerate(br):
                attn_rows += [dict(patient_id=p, branch=g, image_id=im, weight=float(w)) for im, w in zip(ids, a[b])]
        return res, pd.DataFrame(pred_rows), pd.DataFrame(attn_rows)


def config_hash(cfg: dict, spec: RunSpec, profile: dict) -> str:
    """Hash of everything that determines a run besides the spec itself."""
    keys = ("view_dropout", "consistency", "clinical", "protocol")
    blob = dict(variant=variant_config(cfg, spec.variant, profile), **{k: cfg[k] for k in keys},
                reps=profile.get("missing_view_reps"), max_train=profile.get("max_train_patients"))
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run(cfg: dict, co: Cohort, feats: np.ndarray, spec: RunSpec, device: torch.device, profile: dict,
        profile_name: str, force: bool = False) -> dict | None:
    P = cfg["paths"]
    runs_dir = P["outputs"] / "runs"
    rec_p = runs_dir / f"{spec.run_id}.json"
    chash = config_hash(cfg, spec, profile)
    if rec_p.exists() and not force:
        old_hash = json.loads(rec_p.read_text()).get("config_hash")
        if old_hash != chash:
            raise RuntimeError(f"{spec.run_id}: record exists with a different configuration "
                               f"({old_hash} != {chash}); re-run with --force (old record is archived)")
        log.info("skip %s (record exists)", spec.run_id)
        return None
    if rec_p.exists():  # --force: keep the superseded record instead of silently overwriting it
        old = runs_dir / "_superseded"
        old.mkdir(parents=True, exist_ok=True)
        rec_p.rename(old / f"{spec.run_id}.{int(time.time())}.json")
    t0 = time.time()
    r = Runner(cfg, co, feats, spec, device, profile, P["outputs"])
    fit = r.fit()
    return finish(cfg, r, fit, profile, profile_name, chash, t0)


def finish(cfg: dict, r: Runner, fit: dict, profile: dict, profile_name: str, chash: str, t0: float) -> dict:
    """Calibrate, evaluate at all image fractions, and write checkpoint/predictions/record."""
    P, spec, device = cfg["paths"], r.spec, r.device
    rec_p = P["outputs"] / "runs" / f"{spec.run_id}.json"
    fracs = cfg["protocol"]["missing_view_fractions"]
    metrics, preds, attn = r.evaluate(fracs, profile.get("missing_view_reps", 5))
    for d in ("runs", "predictions", "attention", "history"):
        (P["outputs"] / d).mkdir(parents=True, exist_ok=True)
    P["checkpoints"].mkdir(exist_ok=True)
    ck = P["checkpoints"] / f"{spec.run_id}.pt"
    torch.save(dict(state=r.model.state_dict(), spec=asdict(spec), variant=r.v, head=r.head,
                    mapping=r.mapping_np, temperatures=metrics["temperatures"]), ck)
    preds.to_csv(P["outputs"] / "predictions" / f"{spec.run_id}.csv.gz", index=False)
    if len(attn):
        attn.to_csv(P["outputs"] / "attention" / f"{spec.run_id}.csv.gz", index=False)
    pd.DataFrame(fit.pop("history")).to_csv(P["outputs"] / "history" / f"{spec.run_id}.csv", index=False)
    rec = dict(run_id=spec.run_id, **asdict(spec), profile=profile_name, config_hash=chash, variant_config=r.v, head_used=r.head,
               mapping={k: m.tolist() for k, m in (r.mapping_np or {}).items()},
               n_train=len(r.ids["train"]), n_val=len(r.ids["val"]), n_test=len(r.ids["test"]),
               n_params=sum(p.numel() for p in r.model.parameters()) + fit.pop("encoder_params", 0),
               device=str(device), **fit,
               total_seconds=time.time() - t0, checkpoint=str(ck.relative_to(P["checkpoints"].parent)),
               metrics=metrics, env=environment(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    rec_p.write_text(json.dumps(rec, default=float))
    tm = metrics[f"test|1.0|0|cal"]
    with open(P["outputs"] / "experiments.jsonl", "a") as f:
        f.write(json.dumps(dict(run_id=spec.run_id, profile=profile_name, finished=rec["finished"],
                                best_epoch=fit["best_epoch"], seconds=round(rec["total_seconds"], 1),
                                heldout_dist_macro_f1=tm["distribution"]["macro_f1"],
                                heldout_mean_macro_f1=tm["mean"]["macro_f1"])) + "\n")
    log.info("%s: best epoch %d, held-out dist macro-F1 %.3f, mean macro-F1 %.3f (%.0fs)", spec.run_id,
             fit["best_epoch"], tm["distribution"]["macro_f1"], tm["mean"]["macro_f1"], rec["total_seconds"])
    return rec
