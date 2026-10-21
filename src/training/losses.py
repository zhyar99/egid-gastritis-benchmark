"""L_total = L_supervised + lambda1 * L_clinical + lambda2 * L_cross_view."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.metadata import TASK_KEYS


def class_weights(y_train: np.ndarray, n_classes: dict[str, int]) -> dict[str, torch.Tensor]:
    """Inverse-frequency weights from training PATIENTS, normalised to mean 1."""
    out = {}
    for j, t in enumerate(TASK_KEYS):
        y = y_train[:, j][y_train[:, j] >= 0]
        cnt = np.bincount(y, minlength=n_classes[t]).astype(np.float32)
        w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / n_classes[t], 0.0)
        out[t] = torch.as_tensor(w / w[cnt > 0].mean(), dtype=torch.float32)
    return out


def supervised_loss(logp: dict[str, torch.Tensor], y: torch.Tensor, kind: str = "ce",
                    weights: dict[str, torch.Tensor] | None = None, gamma: float = 2.0) -> tuple[torch.Tensor, dict]:
    """Mean over tasks of the per-task loss; missing labels (-1) are masked, never imputed."""
    per_task = {}
    for j, t in enumerate(TASK_KEYS):
        lab = y[:, j]
        ok = lab >= 0
        if not ok.any():
            continue
        lp = logp[t][ok]
        nll = -lp.gather(1, lab[ok, None]).squeeze(1)
        if kind == "focal":
            nll = (1 - (-nll).exp()).pow(gamma) * nll
        if kind == "weighted_ce":
            w = weights[t].to(lp)[lab[ok]]
            per_task[t] = (w * nll).sum() / w.sum()
        else:
            per_task[t] = nll.mean()
    return torch.stack(list(per_task.values())).mean(), {t: v.item() for t, v in per_task.items()}


def cross_view_consistency(teacher_logits: dict[str, torch.Tensor], student_logits: dict[str, torch.Tensor],
                           temperature: float = 2.0, confidence: float = 0.7) -> tuple[torch.Tensor, float]:
    """Confidence-gated, temperature-scaled KL(teacher || student) per output group.

    The teacher (larger view) is detached: it is a target, not a path for
    gradients. Patients whose teacher max-probability (at T=1) is below
    `confidence` contribute nothing. Normalised by batch size (FixMatch-style),
    scaled by T^2 to keep gradient magnitude comparable across temperatures.
    """
    losses, gates = [], []
    for g, zt in teacher_logits.items():
        zt = zt.detach()
        gate = (zt.softmax(-1).max(-1).values >= confidence).float()
        kl = F.kl_div(F.log_softmax(student_logits[g] / temperature, -1), F.log_softmax(zt / temperature, -1),
                      log_target=True, reduction="none").sum(-1)
        losses.append((gate * kl).mean() * temperature ** 2)
        gates.append(gate.mean())
    return torch.stack(losses).mean(), torch.stack(gates).mean().item()


def warmup(epoch: int, start: int, ramp: int) -> float:
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start + 1) / max(ramp, 1))
