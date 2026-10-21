"""Patient-level classification and calibration metrics.

Undefined metrics (e.g. AUROC when a class is absent from the evaluated
patients) are returned as NaN and flagged, never imputed.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def ece(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width bins."""
    conf, pred = P.max(1), P.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def reliability(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf, pred = P.max(1), P.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return [dict(bin=b, lo=b / n_bins, hi=(b + 1) / n_bins, n=int((bins == b).sum()),
                 conf=float(conf[bins == b].mean()) if (bins == b).any() else np.nan,
                 acc=float((pred[bins == b] == y[bins == b]).mean()) if (bins == b).any() else np.nan)
            for b in range(n_bins)]


def task_metrics(y: np.ndarray, P: np.ndarray, class_names: list[str] | None = None) -> dict:
    C = P.shape[1]
    labels = np.arange(C)
    pred = P.argmax(1)
    present = np.unique(y)
    names = class_names or [str(c) for c in labels]
    m = dict(n=int(len(y)), accuracy=accuracy_score(y, pred), balanced_accuracy=np.nan,
             precision_macro=precision_score(y, pred, labels=labels, average="macro", zero_division=0),
             recall_macro=recall_score(y, pred, labels=labels, average="macro", zero_division=0),
             macro_f1=f1_score(y, pred, labels=labels, average="macro", zero_division=0),
             weighted_f1=f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
             nll=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()),
             brier=float(((P - np.eye(C)[y]) ** 2).sum(1).mean()), ece=ece(y, P),
             confidence_mean=float(P.max(1).mean()),
             entropy_mean=float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean()),
             classes_absent=[names[c] for c in labels if c not in present])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    f1s = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    for c in labels:
        # per-class recall of an absent class is undefined
        m[f"f1[{names[c]}]"] = float(f1s[c]) if c in present or (pred == c).any() else np.nan
        m[f"recall[{names[c]}]"] = float(rec[c]) if c in present else np.nan
        m[f"precision[{names[c]}]"] = float(prec[c]) if (pred == c).any() else np.nan
    if len(present) < C:
        m["auroc"] = m["auprc"] = np.nan  # undefined: class absent
    elif C == 2:
        m["auroc"] = roc_auc_score(y, P[:, 1])
        m["auprc"] = average_precision_score(y, P[:, 1])
    else:
        m["auroc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
        m["auprc"] = float(np.mean([average_precision_score(y == c, P[:, c]) for c in labels]))
    m["confusion"] = confusion_matrix(y, pred, labels=labels).tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def all_task_metrics(y: np.ndarray, probs: dict[str, np.ndarray], tasks: list, mapping=None) -> dict:
    """y [P, 4] in TASK_KEYS order; probs: task -> [P, C]."""
    out = {}
    for j, t in enumerate(tasks):
        ok = y[:, j] >= 0
        out[t.key] = task_metrics(y[ok, j], probs[t.key][ok], t.classes)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "nll", "ece"):
        out.setdefault("mean", {})[k] = float(np.nanmean([out[t.key][k] for t in tasks]))
    if mapping is not None:
        from ..models.structured_head import coherence
        pred = {t.key: probs[t.key].argmax(1) for t in tasks}
        out["mean"]["coherence"] = float(coherence(pred, mapping).mean())
    return out
