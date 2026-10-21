"""Hyper-parameter selection on INNER-VALIDATION patients only.

For every config in the grid, fit on each CV fold's training patients and
record the best inner-validation NLL (the early-stopping criterion). The CV
held-out folds and the primary test set are never evaluated here.
Selection: image-level config by IMG; one shared config for all MIL variants
by the mean of the A0 and A1 validation NLLs.

Run one process per variant in parallel, then aggregate:
  python scripts/tune.py --variants IMG &  python scripts/tune.py --variants A0 &  python scripts/tune.py --variants A1
  python scripts/tune.py --select
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.training.feature_extraction import load_features  # noqa: E402
from src.training.trainer import Runner, RunSpec  # noqa: E402

GRID = dict(lr=[1e-4, 3e-4], dropout=[0.25, 0.5], weight_decay=[1e-2, 1e-1], hidden=[128, 256],
            input_dropout=[0.0, 0.3])
KEYS = list(GRID)


def tune_variant(cfg, co, feats, v, encoder, folds, max_epochs, out_p):
    # rows already computed by any worker for this variant are skipped
    prev = [pd.read_csv(p) for p in out_p.parent.glob(f"grid_{encoder}_{v}*.csv")]
    seen = {(r["fold"], *[float(r[k]) for k in KEYS]) for d in prev for r in d.to_dict("records")}
    rows = pd.read_csv(out_p).to_dict("records") if out_p.exists() else []
    cfg_t = {**cfg, "tuned": {}}  # grid values must not be overridden by an earlier selection
    for combo in itertools.product(*GRID.values()):
        hp = dict(zip(KEYS, combo))
        for f in folds:
            if (f, *[float(x) for x in combo]) in seen:
                continue
            prof = dict(overrides=hp, max_epochs=max_epochs)
            r = Runner(cfg_t, co, feats, RunSpec(v, encoder, f, 0), torch.device("cpu"), prof, cfg["paths"]["outputs"])
            fit = r.fit()
            rows.append(dict(variant=v, fold=f, **hp, val_nll=fit["best_val_nll"], best_epoch=fit["best_epoch"],
                             epochs_run=fit["epochs_run"]))
            pd.DataFrame(rows).to_csv(out_p, index=False)


def select(tdir: Path, encoder: str, n_folds: int) -> dict:
    df = pd.concat([pd.read_csv(p) for p in tdir.glob(f"grid_{encoder}_*.csv") if "summary" not in p.name])
    df = df.drop_duplicates(["variant", "fold"] + KEYS)
    agg = df.groupby(["variant"] + KEYS).agg(val_nll=("val_nll", "mean"), sd=("val_nll", "std"), n=("val_nll", "size"),
                                             best_epoch=("best_epoch", "mean")).reset_index()
    agg = agg[agg.n == n_folds]
    agg.to_csv(tdir / f"grid_summary_{encoder}.csv", index=False)
    img = agg[agg.variant == "IMG"].sort_values("val_nll").iloc[0]
    mil = agg[agg.variant.isin(["A0", "A1"])].groupby(KEYS).val_nll.agg(["mean", "size"])
    mil = mil[mil["size"] == 2].sort_values("mean")
    cast = lambda k, x: int(x) if k == "hidden" else float(x)  # noqa: E731
    sel = dict(image={k: cast(k, img[k]) for k in KEYS}, mil=dict(zip(KEYS, [cast(k, x) for k, x in zip(KEYS, mil.index[0])])),
               criterion="mean inner-validation NLL over 5 CV folds (seed 0)")
    # Addendum B: one selection per aggregation family, on its independent-head variant
    for v in ("N1", "N3", "N5"):
        fam = agg[agg.variant == v].sort_values("val_nll")
        if len(fam):
            sel[v] = {k: cast(k, fam.iloc[0][k]) for k in KEYS} | {"val_nll": float(fam.iloc[0].val_nll)}
    (tdir / f"selected_{encoder}.json").write_text(json.dumps(sel, indent=1))
    print(agg.sort_values(["variant", "val_nll"]).groupby("variant").head(5).to_string(index=False))
    return sel


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["IMG", "A0", "A1"])
    ap.add_argument("--encoder", default=cfg["protocol"]["primary_encoder"])
    ap.add_argument("--folds", nargs="*", default=[f"cv{i}" for i in range(5)])
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--select", action="store_true", help="only aggregate the grid files and write the selection")
    a = ap.parse_args()
    setup_logging("tune")
    torch.set_num_threads(a.threads)
    P = cfg["paths"]
    tdir = P["outputs"] / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)
    if not a.select:
        co = Cohort.load(P["data"], P["audit"])
        feats = load_features(P["cache"], a.encoder, 224, co.index.image_id.tolist())
        for v in a.variants:
            suffix = "" if len(a.folds) == 5 else "_" + "-".join(a.folds)
            tune_variant(cfg, co, feats, v, a.encoder, a.folds, a.max_epochs, tdir / f"grid_{a.encoder}_{v}{suffix}.csv")
    else:
        print("selected:", select(tdir, a.encoder, 5))
