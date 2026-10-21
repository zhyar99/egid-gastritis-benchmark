"""Reproducible random image-subset ('missing-view') evaluation.

Metadata carries no anatomical location per image, so this is random
image-subset robustness, not anatomical missing-view robustness. The subset
for (patient, fraction, repetition) depends only on those three values, so
every model is evaluated on exactly the same images.
"""
from __future__ import annotations

import numpy as np


def eval_subset(patient_id: str, n: int, frac: float, rep: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(np.random.SeedSequence([rep, int(round(frac * 1000)), int(patient_id)]))
    k = max(1, int(round(frac * n)))
    return np.sort(rng.choice(n, size=k, replace=False))
