"""Patient-level MIL: mean / gated attention / task-aware gated attention / top-k / transformer pooling.

CoRe-MIL = task-aware attention + structured head + view dropout +
confidence-gated cross-view consistency (the last two live in the trainer).
Addendum B baselines: top-k pooling (HOPE AI), ACMIL-style stochastic top-instance
masking of attention during training, and a TransMIL-style transformer aggregator.
"""
from __future__ import annotations

import torch
from torch import nn

from .structured_head import INDEPENDENT_GROUPS, finalize


class GatedAttention(nn.Module):
    """Ilse et al. (2018) gated attention with `n_branches` independent branches.

    With mask_topk > 0, training mode masks each of the mask_topk highest-scoring images with
    probability mask_p (ACMIL-style), always keeping at least one valid image; inference is unmasked.
    """

    def __init__(self, dim: int, attn_dim: int, n_branches: int, mask_topk: int = 0, mask_p: float = 0.5):
        super().__init__()
        self.n, self.mask_topk, self.mask_p = n_branches, mask_topk, mask_p
        self.V = nn.Linear(dim, attn_dim * n_branches)
        self.U = nn.Linear(dim, attn_dim * n_branches)
        self.w = nn.Parameter(torch.randn(n_branches, attn_dim) * attn_dim ** -0.5)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, K, _ = h.shape
        g = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # [B, K, n*a]
        scores = torch.einsum("bkna,na->bnk", g.view(B, K, self.n, -1), self.w)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if self.training and self.mask_topk > 0:
            k = min(self.mask_topk, K)
            top = scores.topk(k, dim=-1).indices  # [B, n, k], ranked by score
            drop = torch.rand(B, self.n, k, device=h.device) < self.mask_p
            valid = mask.sum(1).view(B, 1, 1)
            drop &= torch.arange(k, device=h.device).view(1, 1, k) < valid - 1  # never mask the last valid image
            out = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top, drop)
            scores = scores.masked_fill(out, float("-inf"))
        return torch.softmax(scores, dim=-1)  # [B, n, K]; exactly 0 on padding


def topk_mean(logits: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Per class, mean of the k largest instance logits over the valid images ([B, K, C] -> [B, C])."""
    x = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    s = x.sort(dim=1, descending=True).values.nan_to_num(neginf=0.0)
    ke = mask.sum(1).clamp(max=k).clamp(min=1)  # k_eff = min(k, N_i)
    csum = s.cumsum(1)
    return csum.gather(1, (ke - 1).view(-1, 1, 1).expand(-1, 1, s.shape[2])).squeeze(1) / ke.view(-1, 1)


class MILModel(nn.Module):
    def __init__(self, in_dim: int, n_classes: dict[str, int], pooling: str = "task_attention",
                 head: str = "independent", mapping: dict | None = None, hidden: int = 256,
                 attn_dim: int = 128, dropout: float = 0.25, input_dropout: float = 0.0, in_norm: bool = False,
                 topk: int = 7, mask_topk: int = 0, mask_p: float = 0.5, tr_layers: int = 2, tr_heads: int = 4, tr_ffn: int = 512):
        super().__init__()
        if head == "structured" and not mapping:
            raise ValueError("structured head needs a verified deterministic mapping")
        self.pooling, self.head, self.topk = pooling, head, topk
        self.groups = INDEPENDENT_GROUPS if head == "structured" else list(n_classes)
        self.embed = nn.Sequential(nn.Dropout(input_dropout), nn.LayerNorm(in_dim) if in_norm else nn.Identity(),
                                   nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        n_br = {"attention": 1, "task_attention": len(self.groups)}.get(pooling, 0)
        if pooling not in ("mean", "attention", "task_attention", "topk", "transformer"):
            raise ValueError(f"unknown pooling {pooling}")
        self.attn = GatedAttention(hidden, attn_dim, n_br, mask_topk, mask_p) if n_br else None
        if pooling == "transformer":
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(hidden, tr_heads, dim_feedforward=tr_ffn, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, tr_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleDict({f"h_{g}": nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes[g]))
                                    for g in self.groups})
        self.mapping_keys = list(mapping) if head == "structured" else []
        for t in self.mapping_keys:
            self.register_buffer(f"M_{t}", torch.as_tensor(mapping[t]))

    @property
    def mapping(self) -> dict[str, torch.Tensor] | None:
        return {t: getattr(self, f"M_{t}") for t in self.mapping_keys} or None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        h = self.embed(x)
        attn = None
        if self.pooling == "topk":  # instance-level logits, top-k mean per class (Hu et al. 2025)
            logits = {g: topk_mean(self.heads[f"h_{g}"](h), mask, self.topk) for g in self.groups}
            return dict(logits=logits, attn=None, h=h)
        if self.pooling == "transformer":  # class token over the unordered image set
            t = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
            pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
            z = self.norm(self.encoder(t, src_key_padding_mask=pad)[:, :1])
        elif self.attn is None:
            m = mask.unsqueeze(-1).to(h)
            z = ((h * m).sum(1) / m.sum(1).clamp_min(1)).unsqueeze(1)  # [B, 1, H]
        else:
            attn = self.attn(h, mask)
            z = attn @ h  # [B, n_br, H]
        logits = {g: self.heads[f"h_{g}"](z[:, i if z.shape[1] > 1 else 0]) for i, g in enumerate(self.groups)}
        return dict(logits=logits, attn=attn, h=h)  # h: image embeddings [B, K, H] (Addendum C)

    def instance_logits(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-image logits from the patient heads (shared weights), for image-level supervision."""
        return {g: self.heads[f"h_{g}"](h) for g in self.groups}

    def log_probs(self, out: dict, temps: dict | None = None) -> dict[str, torch.Tensor]:
        return finalize(out["logits"], self.mapping, temps)
