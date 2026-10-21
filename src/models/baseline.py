"""Image-level multitask baseline (EGID protocol): shared features, four heads,
patient prediction = mean of image softmax probabilities."""
from __future__ import annotations

import torch
from torch import nn

from .structured_head import finalize


class ImageLevelModel(nn.Module):
    pooling, head, mapping = "image_mean", "independent", None

    def __init__(self, in_dim: int, n_classes: dict[str, int], hidden: int = 256, dropout: float = 0.25, input_dropout: float = 0.0,
                 in_norm: bool = False, **_):
        super().__init__()
        self.groups = list(n_classes)
        if hidden:
            self.embed = nn.Sequential(nn.Dropout(input_dropout), nn.LayerNorm(in_dim) if in_norm else nn.Identity(),
                                       nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        else:  # hidden=0: plain linear heads on the backbone feature (EGID paper architecture)
            self.embed, hidden = nn.Identity(), in_dim
        self.heads = nn.ModuleDict({f"h_{g}": nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, c))
                                    for g, c in n_classes.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(x)
        return {g: self.heads[f"h_{g}"](h) for g in self.groups}

    @staticmethod
    def aggregate(image_logits: dict[str, torch.Tensor], patient_pos: torch.Tensor, n_patients: int) -> dict:
        """Patient log of mean image probability, per task (used as patient 'logits')."""
        out = {}
        cnt = torch.zeros(n_patients, device=patient_pos.device).index_add_(
            0, patient_pos, torch.ones_like(patient_pos, dtype=torch.float))
        for g, z in image_logits.items():
            p = torch.zeros(n_patients, z.shape[1], device=z.device).index_add_(0, patient_pos, z.softmax(-1))
            out[g] = torch.log((p / cnt[:, None]).clamp_min(1e-12))
        return out

    def log_probs(self, out: dict, temps: dict | None = None) -> dict[str, torch.Tensor]:
        # softmax(log p_mean / T): T=1 returns the mean probabilities unchanged
        return finalize(out["logits"], None, temps)
