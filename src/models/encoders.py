"""Pretrained image encoders (timm), returning pooled feature vectors."""
from __future__ import annotations

import timm
import torch
from torch import nn


def create_encoder(name: str, enc_cfg: dict, pretrained: bool = True) -> tuple[nn.Module, dict]:
    """Encoder with num_classes=0 (pooled features) and its normalisation config."""
    spec = enc_cfg[name]
    model = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=0, **spec.get("kwargs", {}))
    data_cfg = timm.data.resolve_data_config({}, model=model)
    return model, dict(mean=list(data_cfg["mean"]), std=list(data_cfg["std"]), dim=model.num_features,
                       timm=spec["timm"], n_params=sum(p.numel() for p in model.parameters()))


class Normalize(nn.Module):
    """uint8 NHWC batch -> normalised float NCHW (done on device to keep host copies small)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.permute(0, 3, 1, 2).float()
        return (x - self.mean) / self.std
