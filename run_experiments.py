"""Run variant x encoder x split x seed grids on cached features. Resumable.

Examples:
  python run_experiments.py --profile smoke --group all
  python run_experiments.py --profile dev --group ablations
  python run_experiments.py --profile encoders --group encoders
  python run_experiments.py --profile full --group all --allow-test   # primary test set: once, after protocol freeze
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import get_device, load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.training.feature_extraction import load_features  # noqa: E402
from src.training.trainer import RunSpec, run  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", choices=list(cfg["profiles"]))
    ap.add_argument("--group", default="ablations", choices=list(cfg["groups"]))
    ap.add_argument("--variants", nargs="*", help="override the group's variant list")
    ap.add_argument("--encoders", nargs="*")
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default=cfg["mil_device"])
    ap.add_argument("--allow-test", action="store_true", help="permit the primary split (untouched test set)")
    ap.add_argument("--force", action="store_true", help="re-run existing records (old ones are archived)")
    a = ap.parse_args()
    setup_logging(f"run_{a.profile}")
    log = logging.getLogger("run_experiments")
    prof = dict(cfg["profiles"][a.profile])
    splits = a.splits or prof["splits"]
    if "primary" in splits and not a.allow_test:
        sys.exit("The primary split evaluates the untouched test set. Freeze the protocol, then pass --allow-test.")
    torch.set_num_threads(cfg["torch_threads"])
    dev = get_device(a.device)
    P = cfg["paths"]
    co = Cohort.load(P["data"], P["audit"])
    ids = co.index.image_id.tolist()
    if a.profile == "smoke":  # smoke runs are written to a separate tree
        cfg["paths"] = P = {**P, "outputs": P["outputs"] / "smoke", "checkpoints": P["checkpoints"] / "smoke"}
    variants = a.variants or cfg["groups"][a.group]
    jobs = [(e, v, s, sd) for e in (a.encoders or prof["encoders"]) for s in splits
            for sd in (a.seeds if a.seeds is not None else prof["seeds"]) for v in variants]
    log.info("%d runs on %s (profile %s)", len(jobs), dev, a.profile)
    t0, feats_cache = time.time(), {}
    for i, (enc, var, split, seed) in enumerate(jobs):
        if enc not in feats_cache:
            feats_cache = {enc: load_features(cfg["paths"]["cache"], enc, a.size, ids)}
        run(cfg, co, feats_cache[enc], RunSpec(var, enc, split, seed, a.size), dev, prof, a.profile, a.force)
        log.info("[%d/%d] elapsed %.1f min", i + 1, len(jobs), (time.time() - t0) / 60)
