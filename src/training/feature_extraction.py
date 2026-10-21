"""Frozen-backbone embedding extraction to disk (one float32 row per image)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from ..common import mps_oom_hint
from ..models.encoders import Normalize, create_encoder

log = logging.getLogger(__name__)


def feature_path(cache_dir: Path, encoder: str, size: int) -> Path:
    return Path(cache_dir) / "features" / f"{encoder}_{size}.npy"


@torch.no_grad()
def extract(images: np.ndarray, image_ids: list[str], encoder: str, enc_cfg: dict, size: int,
            cache_dir: Path, device: torch.device, batch_size: int = 32, hflip_tta: bool = False) -> Path:
    out = feature_path(cache_dir, encoder, size)
    meta_p = out.with_suffix(".json")
    if out.exists() and meta_p.exists() and json.loads(meta_p.read_text())["image_ids"] == image_ids:
        log.info("features %s up to date", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    model, info = create_encoder(encoder, enc_cfg)
    model = model.eval().to(device)
    norm = Normalize(info["mean"], info["std"]).to(device)
    feats = np.zeros((len(images), info["dim"]), np.float32)
    t0 = time.time()
    for s in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[s: s + batch_size])).to(device)
        x = norm(x)
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        try:
            f = model(x)
            if hflip_tta:
                f = (f + model(torch.flip(x, dims=[3]))) / 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                raise RuntimeError(mps_oom_hint(e)) from e
            raise
        feats[s: s + len(x)] = f.float().cpu().numpy()
        if s % (batch_size * 40) == 0:
            log.info("%s: %d/%d images, %.1f img/s, RSS %.1f GB", encoder, s + len(x), len(images),
                     (s + len(x)) / (time.time() - t0), psutil.Process().memory_info().rss / 2**30)
    if not np.isfinite(feats).all():
        raise FloatingPointError(f"non-finite features from {encoder} on {device}")
    np.save(out, feats)
    meta_p.write_text(json.dumps(dict(encoder=encoder, size=size, device=str(device), seconds=time.time() - t0,
                                      hflip_tta=hflip_tta, **info, image_ids=image_ids)))
    log.info("%s done in %.0fs -> %s %s", encoder, time.time() - t0, out, feats.shape)
    return out


def load_features(cache_dir: Path, encoder: str, size: int, image_ids: list[str]) -> np.ndarray:
    p = feature_path(cache_dir, encoder, size)
    if json.loads(p.with_suffix(".json").read_text())["image_ids"] != image_ids:
        raise ValueError(f"{p}: image order differs from the index; re-run scripts/extract_features.py")
    return np.load(p)


def aug_feature_path(cache_dir: Path, encoder: str, size: int, n_views: int) -> Path:
    return Path(cache_dir) / "features" / f"{encoder}_{size}_aug{n_views}.npy"


@torch.no_grad()
def extract_augmented(images: np.ndarray, image_ids: list[str], encoder: str, enc_cfg: dict, size: int,
                      cache_dir: Path, device: torch.device, n_views: int, batch_size: int = 32) -> Path:
    """n_views augmented copies of every image (fine-tuning augmentation, seed = view index) -> [n_views, N, D]."""
    from .finetune import transforms
    out = aug_feature_path(cache_dir, encoder, size, n_views)
    meta_p = out.with_suffix(".json")
    if out.exists() and meta_p.exists() and json.loads(meta_p.read_text())["image_ids"] == image_ids:
        log.info("augmented features %s up to date", out)
        return out
    model, info = create_encoder(encoder, enc_cfg)
    model = model.eval().to(device)
    norm = Normalize(info["mean"], info["std"]).to(device)
    aug = transforms(size, True)
    feats = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), "w+", np.float32, (n_views, len(images), info["dim"]))
    t0 = time.time()
    for k in range(n_views):
        torch.manual_seed(k)
        for s in range(0, len(images), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(images[s: s + batch_size])).permute(0, 3, 1, 2)
            x = torch.stack([aug(im) for im in x]).permute(0, 2, 3, 1)  # uint8 NHWC, size x size
            feats[k, s: s + len(x)] = model(norm(x.to(device))).float().cpu().numpy()
        log.info("%s augmented view %d/%d done (%.0fs)", encoder, k + 1, n_views, time.time() - t0)
    if not np.isfinite(feats).all():
        raise FloatingPointError(f"non-finite augmented features from {encoder} on {device}")
    feats.flush()
    del feats
    out.with_suffix(".tmp.npy").rename(out)
    meta_p.write_text(json.dumps(dict(encoder=encoder, size=size, n_views=n_views, device=str(device),
                                      seconds=time.time() - t0, augmentation=repr(aug), **info, image_ids=image_ids)))
    return out


def load_aug_features(cache_dir: Path, encoder: str, size: int, n_views: int, image_ids: list[str]) -> np.ndarray:
    p = aug_feature_path(cache_dir, encoder, size, n_views)
    if json.loads(p.with_suffix(".json").read_text())["image_ids"] != image_ids:
        raise ValueError(f"{p}: image order differs from the index; re-run scripts/extract_features.py --aug-views")
    return np.load(p)
