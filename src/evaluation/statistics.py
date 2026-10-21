"""Patient-level resampling statistics. Patients, never images, are resampled."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def macro_f1(y: np.ndarray, P: np.ndarray) -> float:
    return f1_score(y, P.argmax(1), labels=np.arange(P.shape[1]), average="macro", zero_division=0)


def bootstrap_ci(y: np.ndarray, P: np.ndarray, fn: Callable = macro_f1, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.array([fn(y[i], P[i]) for i in (rng.integers(n, size=n) for _ in range(n_boot))])
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return dict(estimate=float(fn(y, P)), lo=float(lo), hi=float(hi), se=float(np.nanstd(stats)))


def paired_bootstrap(y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray, fn: Callable = macro_f1, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Difference fn(A) - fn(B) on the same resampled patients."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(n, size=n)
        d.append(fn(y[i], Pa[i]) - fn(y[i], Pb[i]))
    d = np.array(d)
    est = fn(y, Pa) - fn(y, Pb)
    lo, hi = np.quantile(d, [0.025, 0.975])
    # two-sided bootstrap p-value: how often the resampled difference falls on the other side of 0
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(diff=float(est), lo=float(lo), hi=float(hi), p_boot=p, effect_size=float(est / (d.std() + 1e-12)))


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on per-patient correctness disagreements."""
    a, b = pred_a == y, pred_b == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only_correct=n01, b_only_correct=n10, p_exact=float(p))


def seed_summary(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))
