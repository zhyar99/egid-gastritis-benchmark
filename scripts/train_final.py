"""Train a final model (default CoRe-MIL A6; or e.g. --variant IMG --encoder densenet121 for the image-level
CNN) on ALL 229 patients and save one self-contained model file.

No patient is held out, so nothing here is an evaluation. Settings that normally come from validation
data are fixed from the cross-validation runs of the same configuration:
  - epochs      = median best epoch of the A6 CV runs + 1
  - temperature = median validation-fitted temperature of the A6 CV runs (per output group)
The expected performance of this model is the CV estimate reported in research/technical_report.md.

  python scripts/train_final.py
  python scripts/train_final.py --variant IMG --encoder densenet121 --out models/cnn_densenet121_final.pt
  python scripts/train_final.py --variant FT_A6 --out models/coremil_ft_final.pt     # all layers trained (MPS)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import environment, get_device, load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.models.encoders import create_encoder  # noqa: E402
from src.training.feature_extraction import load_features  # noqa: E402
from src.training.trainer import Runner, RunSpec, config_hash  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A6")
    ap.add_argument("--encoder", default=cfg["protocol"]["primary_encoder"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/coremil_final.pt")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 8 patients (pipeline check only)")
    a = ap.parse_args()
    cfg["variants"].update(cfg["finetune_variants"])
    finetune = a.variant in cfg["finetune_variants"]
    setup_logging("train_final")
    P = cfg["paths"]
    torch.set_num_threads(cfg["torch_threads"])
    out = Path(a.out) if Path(a.out).is_absolute() else Path(__file__).resolve().parents[1] / a.out
    if out.exists():
        sys.exit(f"{out} exists; move it away first (models are never silently overwritten)")

    # settings fixed from the CV runs of the identical configuration
    # same profile as the CV runs, so the configuration hash matches theirs
    prof = dict(missing_view_reps=10) if finetune else dict(cfg["profiles"]["full"])
    chash = config_hash(cfg, RunSpec(a.variant, a.encoder, "cv0", 0), prof)
    cv = [json.loads(p.read_text()) for p in (P["outputs"] / "runs").glob(f"{a.variant}__{a.encoder}_224__cv*__s*.json")]
    cv = [r for r in cv if r["config_hash"] == chash]
    if len(cv) < 5:
        sys.exit(f"need the CV runs of {a.variant}/{a.encoder} with the current configuration first")
    epochs = 1 if a.smoke else int(np.median([r["best_epoch"] for r in cv])) + 1
    groups = list(cv[0]["metrics"]["temperatures"])
    temps = {g: float(np.median([r["metrics"]["temperatures"][g] for r in cv])) for g in groups}

    co = Cohort.load(P["data"], P["audit"])
    all_ids = sorted(co.bags)
    split = dict(name="final_all", seed=a.seed, train=all_ids, val=[], test=[],
                 note="All patients; final model, no held-out evaluation.")
    (P["splits"] / "final_all.json").write_text(json.dumps(split, indent=1))

    t0 = time.time()
    log = logging.getLogger("train_final")
    log.info("training %s/%s: %d epochs on all %d patients", a.variant, a.encoder, epochs, len(all_ids))

    def train(step) -> list[dict]:
        hist = []
        for e in range(epochs):
            hist.append(step(e))
            log.info("%s/%s epoch %d/%d: train loss %.4f (%.0fs)", a.variant, a.encoder, e + 1, epochs, hist[-1]["sup"],
                     time.time() - t0)
        return hist
    if finetune:  # encoder + MIL head trained end-to-end on MPS
        from src.training.finetune import FineTuner
        ftr = FineTuner(cfg, co, RunSpec(a.variant, a.encoder, "final_all", a.seed), get_device(cfg["device"]),
                        dict(max_train_patients=8) if a.smoke else {})
        epoch_fn = (lambda e: ftr.epoch_image()) if ftr.v["kind"] == "image" else ftr.epoch_mil
        hist = train(lambda e: dict(sup=epoch_fn(e)))
        model, head_kind, groups = ftr.head.cpu().eval(), ftr.v.get("head", "independent"), ftr.head.groups
        mapping_np, vcfg, info = ftr.mapping_np, ftr.v, ftr.info
        enc_state = {k: v.cpu() for k, v in ftr.enc.state_dict().items()}
        n_train = len(ftr.ids["train"])
        # fine-tuned heads normalise features themselves (LayerNorm); evaluation resized the 256-px cache
        mu, sd, resize_from, in_norm = torch.zeros(info["dim"]), torch.ones(info["dim"]), ftr.ft["cache_size"], True
    else:  # frozen encoder, head trained on cached features (CPU)
        feats = load_features(P["cache"], a.encoder, 224, co.index.image_id.tolist())
        r = Runner(cfg, co, feats, RunSpec(a.variant, a.encoder, "final_all", a.seed), torch.device("cpu"),
                   prof, P["outputs"])
        r.opt = torch.optim.AdamW(r.model.parameters(), lr=r.v["lr"], weight_decay=r.v["weight_decay"])
        epoch_fn = r._epoch_image if r.v["kind"] == "image" else r._epoch_mil
        hist = train(epoch_fn)
        model, head_kind, groups, mapping_np, vcfg = r.model.eval(), r.head, r.model.groups, r.mapping_np, r.v
        # feature standardisation statistics (training images = all images)
        f = torch.as_tensor(feats)
        mu, sd, resize_from, in_norm = f.mean(0), f.std(0).clamp_min(1e-6), None, False
        enc, info = create_encoder(a.encoder, cfg["encoders"])
        enc_state = enc.state_dict()
        n_train = len(r.ids["train"])
    cv_f1 = [r_["metrics"]["test|1.0|0|cal"]["distribution"]["macro_f1"] for r_ in cv]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        format="coremil-final-v1", kind=vcfg["kind"], finetuned=finetune, in_norm=in_norm, resize_from=resize_from,
        variant=a.variant, variant_config=vcfg, head=head_kind, groups=groups,
        head_state=model.state_dict(), mapping={k: v.tolist() for k, v in (mapping_np or {}).items()},
        n_classes=co.n_classes, classes={t.key: t.classes for t in co.tasks},
        encoder=a.encoder, encoder_timm=info["timm"], encoder_kwargs=cfg["encoders"][a.encoder].get("kwargs", {}),
        encoder_state=enc_state,
        image_size=224, preprocessing="RGB, black-padded to square, bicubic resize to 224"
        + (" via 256 + antialiased bilinear" if resize_from else "") + ", ImageNet mean/std",
        norm_mean=info["mean"], norm_std=info["std"], feat_mean=mu, feat_std=sd,
        temperatures=temps, epochs=epochs, seed=a.seed, n_patients=n_train, n_images=len(co.index),
        settings_from=f"{len(cv)} CV runs (config {chash}): median best epoch +1, median temperatures",
        cv_reference=dict(note="per-fold-seed held-out distribution macro-F1 of the same configuration; "
                               "pooled CV estimate is in research/technical_report.md",
                          median=float(np.median(cv_f1)), runs=len(cv_f1)),
        history=hist, env=environment(), trained=time.strftime("%Y-%m-%d %H:%M:%S"),
        train_seconds=time.time() - t0,
    ), out)
    print(f"saved {out} ({out.stat().st_size / 2**20:.1f} MB): {epochs} epochs on {n_train} patients, "
          f"temperatures {temps}, final train loss {hist[-1]['sup']:.3f}")
