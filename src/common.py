"""Config loading, device selection, seeding and provenance."""
from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("coremil")


def load_config() -> dict:
    cfg = {}
    for name in ("mac_m1.yaml", "experiments.yaml"):
        cfg.update(yaml.safe_load((ROOT / "configs" / name).read_text()))
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        pref = "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "mps":
        if not torch.backends.mps.is_available():
            log.warning("MPS requested but unavailable; using CPU")
            return torch.device("cpu")
        # Report rather than silently hide CPU fallback of unsupported MPS ops.
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            log.warning("PYTORCH_ENABLE_MPS_FALLBACK=1: unsupported MPS ops will run on CPU")
    return torch.device(pref)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str | None:
    """HEAD of this project's own repository (None if the project is not its own repo)."""
    try:
        top = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() != ROOT:
            return None
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src", "configs", "scripts"],
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def code_hash() -> str:
    """Content hash of src/, configs/ and scripts/ (works with or without git)."""
    import hashlib
    h = hashlib.sha1()
    for d in ("src", "configs", "scripts"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix in (".py", ".yaml"):
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
    h.update((ROOT / "run_experiments.py").read_bytes())
    return h.hexdigest()[:12]


def environment() -> dict:
    import sklearn
    import timm
    return dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__,
                numpy=np.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
                machine=platform.machine(), mps=torch.backends.mps.is_available(), git=git_commit(), code_hash=code_hash())


def setup_logging(name: str) -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logs / f"{name}.log")])


def mps_oom_hint(e: Exception) -> str:
    return (f"{e}\nMPS memory error: lower feature_extraction.batch_size / finetune.batch_images or "
            "bag_size_train in configs/mac_m1.yaml, close other GPU-heavy apps, or set "
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to let MPS use more unified memory (may swap).")
