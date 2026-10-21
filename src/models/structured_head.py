"""Structured clinical output: derive dependent tasks from the atrophy-distribution head.

The mapping is re-estimated from the TRAINING patients of each split and only
used when it is exactly deterministic there (no test-set information).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ..data.metadata import TASK_KEYS

log = logging.getLogger(__name__)
SOURCE = "distribution"
INDEPENDENT_GROUPS = ["hp", "distribution"]


def derive_mapping(y_train: np.ndarray, n_classes: dict[str, int], source: str = SOURCE,
                   targets: tuple[str, ...] = ("atrophy", "type")) -> dict[str, np.ndarray] | None:
    """One-hot matrices M_t [C_source, C_t] if every source class maps to exactly one target class."""
    s = y_train[:, TASK_KEYS.index(source)]
    out = {}
    for t in targets:
        yt = y_train[:, TASK_KEYS.index(t)]
        ok = (s >= 0) & (yt >= 0)
        ct = np.zeros((n_classes[source], n_classes[t]), int)
        np.add.at(ct, (s[ok], yt[ok]), 1)
        nz = (ct > 0).sum(1)
        if (nz != 1).any():
            log.warning("%s -> %s not deterministic on training patients (support per source class %s); "
                        "structured head unavailable", source, t, ct.sum(1).tolist())
            return None
        out[t] = (ct > 0).astype(np.float32)
    return out


def finalize(base_logp: dict[str, torch.Tensor], mapping: dict[str, torch.Tensor] | None,
             temps: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    """Per-task log-probabilities from per-group log-probabilities (or logits).

    Temperature is applied per independent group; dependent tasks are pushed
    through the mapping p_t = p_distribution @ M_t, so they are consistent with
    the distribution prediction by construction.
    """
    temps = temps or {}
    out = {g: F.log_softmax(v / temps.get(g, 1.0), dim=-1) for g, v in base_logp.items()}
    if mapping:
        p = out[SOURCE].exp()
        for t, M in mapping.items():
            out[t] = torch.log((p @ M.to(p)).clamp_min(1e-12))
    return out


def soft_clinical_loss(logp: dict[str, torch.Tensor], mapping: dict[str, torch.Tensor]) -> torch.Tensor:
    """For independent heads: KL(p_t || p_distribution @ M_t), averaged over dependent tasks."""
    p_src = logp[SOURCE].exp()
    terms = []
    for t, M in mapping.items():
        q = (p_src @ M.to(p_src)).clamp_min(1e-12)
        terms.append(F.kl_div(q.log(), logp[t], log_target=True, reduction="batchmean"))
    return torch.stack(terms).mean()


def coherence(pred: dict[str, np.ndarray], mapping: dict[str, np.ndarray]) -> np.ndarray:
    """Per-patient flag: are the argmax predictions jointly valid under the mapping?"""
    ok = np.ones(len(pred[SOURCE]), bool)
    for t, M in mapping.items():
        ok &= M[pred[SOURCE], pred[t]] > 0
    return ok
