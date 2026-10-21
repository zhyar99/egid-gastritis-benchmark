"""Optional end-to-end fine-tuning on MPS (needs the 256-px cache: prepare_data.py --sizes 224 256).

  python scripts/finetune.py --variant FT_IMG --encoder densenet121 --split cv0
  python scripts/finetune.py --variant FT_A6 --encoder efficientnet_b0 --split cv0 --smoke
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import get_device, load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.training.finetune import run_finetune  # noqa: E402
from src.training.trainer import RunSpec  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    cfg["variants"].update(cfg["finetune_variants"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(cfg["finetune_variants"]))
    ap.add_argument("--encoder", default=cfg["protocol"]["primary_encoder"])
    ap.add_argument("--splits", nargs="+", default=["cv0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default=cfg["device"])
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 8 training patients, separate output tree")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    setup_logging(f"finetune_{a.variant}")
    if "primary" in a.splits and not a.allow_test:
        sys.exit("The primary split evaluates the untouched test set; pass --allow-test after the protocol freeze.")
    prof = dict(missing_view_reps=10)
    if a.smoke:
        prof.update(max_epochs=2, patience=2, max_train_patients=8, missing_view_reps=2)
        cfg["paths"] = {**cfg["paths"], "outputs": cfg["paths"]["outputs"] / "smoke",
                        "checkpoints": cfg["paths"]["checkpoints"] / "smoke"}
    torch.set_num_threads(cfg["torch_threads"])
    co = Cohort.load(cfg["paths"]["data"], cfg["paths"]["audit"])
    dev = get_device(a.device)
    for split in a.splits:
        for seed in a.seeds:
            run_finetune(cfg, co, RunSpec(a.variant, a.encoder, split, seed, cfg["finetune"]["image_size"]), dev, prof,
                         "smoke" if a.smoke else "finetune", a.force)
