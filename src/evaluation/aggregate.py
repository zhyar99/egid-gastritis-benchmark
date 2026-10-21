"""Aggregate run records/predictions into result tables and statistics.

Schemes
  cv   : out-of-fold predictions of cv0..cv4 pooled -> every development patient (n=195) held out once per seed
  test : primary split, untouched test cohort (n=34)

Point estimate = mean over training seeds of the metric on the pooled
patients. Uncertainty is separated into
  seed_sd      : between-seed SD (model randomness)
  CI (lo, hi)  : patient bootstrap of the seed-averaged metric (same resampled patients for every seed/model)
  rep_sd       : between-subset SD for missing-view evaluation (image sampling)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.metadata import TASK_KEYS
from .metrics import all_task_metrics, reliability, task_metrics
from .statistics import mcnemar

SCHEMES = {"cv": [f"cv{i}" for i in range(5)], "test": ["primary"]}


def load_records(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir).glob("*.json")):
        r = json.loads(p.read_text())
        rows.append(dict(run_id=r["run_id"], variant=r["variant"], encoder=r["encoder"], size=r["size"],
                         split=r["split"], seed=r["seed"], head_used=r["head_used"], n_params=r["n_params"],
                         train_seconds=r["train_seconds"], total_seconds=r["total_seconds"],
                         epochs_run=r["epochs_run"], best_epoch=r["best_epoch"], device=r["device"],
                         temperatures=json.dumps(r["metrics"]["temperatures"]), mapping=json.dumps(r["mapping"]),
                         config_hash=r.get("config_hash")))
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def _read(pred_dir: str, run_id: str) -> pd.DataFrame:
    d = pd.read_csv(Path(pred_dir) / f"{run_id}.csv.gz", dtype={"patient_id": str})
    return d[d.set == "test"]


class Pooled:
    """Pooled held-out predictions of one (variant, encoder, size) under one scheme."""

    def __init__(self, recs: pd.DataFrame, pred_dir: Path, variant: str, encoder: str, size: int, scheme: str):
        sub = recs[(recs.variant == variant) & (recs.encoder == encoder) & (recs["size"] == size) &
                   recs.split.isin(SCHEMES[scheme])]
        need = set(SCHEMES[scheme])
        seeds = [s for s, g in sub.groupby("seed") if set(g.split) == need]  # complete seeds only
        self.seeds = sorted(seeds)
        self.ok = bool(self.seeds)
        if not self.ok:
            return
        parts = []
        for rid in sub[sub.seed.isin(self.seeds)].run_id:
            d = _read(str(pred_dir), rid).copy()
            d["seed"] = int(sub.loc[sub.run_id == rid, "seed"].iloc[0])
            parts.append(d)
        self.df = pd.concat(parts, ignore_index=True)
        self.pids = sorted(self.df.patient_id.unique())
        mp = json.loads(sub.mapping.iloc[0])
        self.mapping = {k: np.array(v) for k, v in mp.items()} or None

    def arrays(self, seed: int, frac: float = 1.0, rep: int = 0, cal: str = "cal") -> tuple[np.ndarray, dict]:
        """(labels [P, 4], {task: probs [P, C]}) in self.pids order; memoised."""
        key = (seed, float(frac), int(rep), cal)
        if not hasattr(self, "_cache"):
            self._cache = {}
            self._groups = {k: g for k, g in self.df.groupby(["seed", "frac", "rep", "cal"])}
        if key not in self._cache:
            d = self._groups[key]
            y = np.zeros((len(self.pids), len(TASK_KEYS)), int)
            P = {}
            for j, t in enumerate(TASK_KEYS):
                dt = d[d.task == t].set_index("patient_id").loc[self.pids]
                y[:, j] = dt.label.to_numpy()
                P[t] = dt[[c for c in dt.columns if c.startswith("p") and c[1:].isdigit()]].dropna(axis=1).to_numpy()
            self._cache[key] = (y, P)
        return self._cache[key]

    def reps(self, frac: float) -> list[int]:
        return sorted(self.df[self.df.frac == frac].rep.unique())


def metric_value(m: dict, task: str, metric: str) -> float:
    return float(m[task][metric]) if task in m and metric in m[task] else np.nan


def summarize(pool: Pooled, tasks, frac: float = 1.0, cal: str = "cal", n_boot: int = 2000, seed: int = 0,
              metrics=("macro_f1", "balanced_accuracy", "accuracy", "auroc", "auprc", "nll", "brier", "ece")) -> list[dict]:
    """Seed-averaged metric per task, seed SD, patient-bootstrap CI and missing-view rep SD."""
    rows = []
    reps = pool.reps(frac) if frac < 1 else [0]
    per = {}  # (seed, rep) -> (y, P)
    for s in pool.seeds:
        for r in reps:
            per[(s, r)] = pool.arrays(s, frac, r, cal)
    mets = {k: all_task_metrics(y, P, tasks, pool.mapping) for k, (y, P) in per.items()}
    rng = np.random.default_rng(seed)
    n = len(pool.pids)
    boots = [rng.integers(n, size=n) for _ in range(n_boot)]
    keys = list(per)
    for t in [x.key for x in tasks] + ["mean"]:
        for metric in metrics + (("coherence",) if t == "mean" else ()):
            vals = np.array([metric_value(mets[k], t, metric) for k in keys])
            if np.all(np.isnan(vals)):
                continue
            by_seed = np.array([np.nanmean([metric_value(mets[(s, r)], t, metric) for r in reps]) for s in pool.seeds])
            by_rep = np.array([np.nanmean([metric_value(mets[(s, r)], t, metric) for s in pool.seeds]) for r in reps])
            row = dict(task=t, metric=metric, frac=frac, cal=cal, estimate=float(np.nanmean(vals)),
                       seed_sd=float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else 0.0,
                       rep_sd=float(np.std(by_rep, ddof=1)) if len(by_rep) > 1 else 0.0,
                       n_patients=n, n_seeds=len(pool.seeds), n_reps=len(reps))
            if t != "mean" and frac >= 1.0 and metric in ("macro_f1", "balanced_accuracy", "accuracy"):
                bs = boot_metric(per, keys, t, metric, boots)
                row.update(lo=float(np.nanquantile(bs, 0.025)), hi=float(np.nanquantile(bs, 0.975)))
            rows.append(row)
    return rows


def _fast_metric(y: np.ndarray, P: np.ndarray, metric: str) -> float:
    """numpy macro-F1 / accuracy / balanced accuracy (same definitions as sklearn with zero_division=0)."""
    C = P.shape[1]
    cm = np.bincount(y * C + P.argmax(1), minlength=C * C).reshape(C, C)
    tp, sup, prd = np.diag(cm), cm.sum(1), cm.sum(0)
    if metric == "accuracy":
        return tp.sum() / len(y)
    if metric == "macro_f1":
        den = sup + prd
        return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())
    if metric == "balanced_accuracy":
        return float((tp[sup > 0] / sup[sup > 0]).mean())
    raise ValueError(metric)


def boot_metric(per: dict, keys: list, task: str, metric: str, boots: list[np.ndarray]) -> np.ndarray:
    """Bootstrap distribution of the (seed x rep)-averaged metric, resampling patients."""
    j = TASK_KEYS.index(task)
    return np.array([np.mean([_fast_metric(per[k][0][i, j], per[k][1][task][i], metric) for k in keys])
                     for i in boots])


def paired(pool_a: Pooled, pool_b: Pooled, task: str = "distribution", metric: str = "macro_f1", frac: float = 1.0,
           n_boot: int = 2000, seed: int = 0) -> dict:
    """A - B on identical patients, seed-averaged; bootstrap CI/p; McNemar on seed-ensembled predictions."""
    if pool_a.pids != pool_b.pids:
        raise ValueError("pooled patient sets differ")
    reps = pool_a.reps(frac) if frac < 1 else [0]
    j = TASK_KEYS.index(task)
    A = {(s, r): pool_a.arrays(s, frac, r) for s in pool_a.seeds for r in reps}
    B = {(s, r): pool_b.arrays(s, frac, r) for s in pool_b.seeds for r in reps}
    rng = np.random.default_rng(seed)
    n = len(pool_a.pids)

    def stat(idx):
        a = np.mean([_fast_metric(y[idx, j], P[task][idx], metric) for y, P in A.values()])
        b = np.mean([_fast_metric(y[idx, j], P[task][idx], metric) for y, P in B.values()])
        return a - b

    full = np.arange(n)
    est = stat(full)
    d = np.array([stat(rng.integers(n, size=n)) for _ in range(n_boot)])
    y = next(iter(A.values()))[0][:, j]
    ens_a = np.mean([P[task] for _, P in A.values()], 0).argmax(1)
    ens_b = np.mean([P[task] for _, P in B.values()], 0).argmax(1)
    mc = mcnemar(y, ens_a, ens_b)
    return dict(task=task, metric=metric, frac=frac, diff=float(est), lo=float(np.quantile(d, 0.025)),
                hi=float(np.quantile(d, 0.975)), p_boot=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                std_effect=float(est / (d.std() + 1e-12)), mcnemar_a_only=mc["a_only_correct"],
                mcnemar_b_only=mc["b_only_correct"], mcnemar_p=mc["p_exact"], n_patients=n,
                n_seeds=min(len(pool_a.seeds), len(pool_b.seeds)))


def per_class(pool: Pooled, tasks, task: str = "distribution") -> pd.DataFrame:
    t = next(x for x in tasks if x.key == task)
    rows = []
    for s in pool.seeds:
        y, P = pool.arrays(s)
        m = task_metrics(y[:, TASK_KEYS.index(task)], P[task], t.classes)
        for c in t.classes:
            rows.append(dict(seed=s, cls=c, f1=m[f"f1[{c}]"], recall=m[f"recall[{c}]"], precision=m[f"precision[{c}]"],
                             support=int((y[:, TASK_KEYS.index(task)] == t.classes.index(c)).sum())))
    return pd.DataFrame(rows).groupby("cls", sort=False).agg(f1=("f1", "mean"), f1_sd=("f1", "std"), recall=("recall", "mean"),
                                                             precision=("precision", "mean"), support=("support", "first")).reset_index()


def confusion(pool: Pooled, task: str = "distribution") -> np.ndarray:
    """Seed-summed confusion matrix / number of seeds (average counts)."""
    j = TASK_KEYS.index(task)
    tot = None
    for s in pool.seeds:
        y, P = pool.arrays(s)
        C = P[task].shape[1]
        cm = np.zeros((C, C))
        np.add.at(cm, (y[:, j], P[task].argmax(1)), 1)
        tot = cm if tot is None else tot + cm
    return tot / len(pool.seeds)


def reliability_table(pool: Pooled, task: str, cal: str) -> pd.DataFrame:
    ys, Ps = [], []
    for s in pool.seeds:
        y, P = pool.arrays(s, cal=cal)
        ys.append(y[:, TASK_KEYS.index(task)])
        Ps.append(P[task])
    return pd.DataFrame(reliability(np.concatenate(ys), np.concatenate(Ps)))
