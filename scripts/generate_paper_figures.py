"""Phase 9: Figures 1-8 from saved outputs (run scripts/evaluate.py first)."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.data.preprocessing import open_cache  # noqa: E402
from src.visualization import architecture, figures as F  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="cv", choices=["cv", "test"])
    ap.add_argument("--smoke", action="store_true", help="read the smoke-test output tree (pipeline check only)")
    a = ap.parse_args()
    P = cfg["paths"]
    if a.smoke:
        P = cfg["paths"] = {**P, "outputs": P["outputs"] / "smoke"}
    out = P["outputs"] / "figures" / a.scheme
    F.style()
    out.mkdir(parents=True, exist_ok=True)
    architecture.draw(str(P["outputs"] / "figures" / "fig1_architecture"))
    F.fig_dataset(P["audit"], P["outputs"] / "figures")
    m = P["outputs"] / "metrics" / a.scheme
    long = pd.read_csv(m / "summary_long.csv")
    co = Cohort.load(P["data"], P["audit"])
    dist_classes = next(t.classes for t in co.tasks if t.key == "distribution")
    F.fig_baselines(long, out, a.scheme)
    F.fig_confusion(json.loads((m / "distribution_confusion.json").read_text()), dist_classes, out)
    F.fig_missing_view(long, out)
    F.fig_ablation(long, out)
    F.fig_calibration(pd.read_csv(m / "reliability_distribution.csv"), long, out)
    # interpretability: full CoRe-MIL, seed 0, held-out patients of the scheme
    splits = ["primary"] if a.scheme == "test" else [f"cv{i}" for i in range(5)]
    enc = cfg["protocol"]["primary_encoder"]
    ids = [f"A6__{enc}_224__{s}__s0" for s in splits]
    att = [pd.read_csv(P["outputs"] / "attention" / f"{i}.csv.gz", dtype={"patient_id": str}) for i in ids
           if (P["outputs"] / "attention" / f"{i}.csv.gz").exists()]
    prd = [pd.read_csv(P["outputs"] / "predictions" / f"{i}.csv.gz", dtype={"patient_id": str}) for i in ids
           if (P["outputs"] / "predictions" / f"{i}.csv.gz").exists()]
    if att:
        att, prd = pd.concat(att), pd.concat(prd)
        prd = prd[prd.set == "test"]
        F.fig_attention(att, prd, open_cache(P["cache"], 224, co.index), co.index, out, dict(enumerate(dist_classes)))
        st = F.attention_stats(att)
        st.to_csv(m / "attention_stats.csv", index=False)
        print(st.describe().T[["mean", "50%", "std"]].round(3).to_string())
    print("figures ->", out)
