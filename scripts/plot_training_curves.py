"""Training vs validation curves of the cross-validation runs (cv0-cv4, all seeds) for the four main models.

  python scripts/plot_training_curves.py   ->  outputs/figures/cv/figS_training_curves.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "outputs" / "history"
OUT = ROOT / "outputs" / "figures" / "cv" / "figS_training_curves"
TRAIN, VAL, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"
MODELS = [("FT_A6__efficientnet_b0", "CoRe-MIL, fine-tuned (your model)", "val_mean_acc", "Validation mean accuracy", "coremil_ft_final.pt"),
          ("A6__efficientnet_b0", "CoRe-MIL, frozen EfficientNet-B0", "val_dist_f1", "Validation distribution macro-F1", None),
          ("IMG__densenet121", "Image-level DenseNet121, frozen", "val_dist_f1", "Validation distribution macro-F1", None),
          ("FT2_IMG__efficientnet_b0", "Image-level EfficientNet-B0, fine-tuned", "val_mean_acc", "Validation mean accuracy", None)]


def mean_curve(runs, col):
    """Mean over the runs still training at each epoch, shown while at least a third of them are."""
    n = max(len(r) for r in runs)
    m = np.full((len(runs), n), np.nan)
    for i, r in enumerate(runs):
        m[i, :len(r)] = r[col].to_numpy()
    keep = (~np.isnan(m)).sum(0) >= len(runs) / 3
    return np.arange(n)[keep], np.nanmean(m, 0)[keep]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(len(MODELS), 2, figsize=(10, 2.6 * len(MODELS)), constrained_layout=True)
for row, (stem, title, vcol, vlabel, final) in zip(axes, MODELS):
    runs = [pd.read_csv(f) for f in sorted(H.glob(f"{stem}_224__cv*__s*.csv"))]
    best = int(np.median([r.val_nll.idxmin() for r in runs]))
    ax = row[0]
    for r in runs:
        ax.plot(r.epoch, r.sup, color=TRAIN, lw=0.6, alpha=0.18)
        ax.plot(r.epoch, r.val_nll, color=VAL, lw=0.6, alpha=0.18)
    ax.plot(*mean_curve(runs, "sup"), color=TRAIN, lw=2, label="Training loss")
    ax.plot(*mean_curve(runs, "val_nll"), color=VAL, lw=2, label="Validation loss (NLL)")
    if final:
        h = torch.load(ROOT / "models" / final, map_location="cpu", weights_only=False)["history"]
        ax.plot(range(len(h)), [e["sup"] for e in h], color=INK, lw=1.4, ls="--", label="Final model, all 229 patients")
    ax.set(title=f"{title}: loss ({len(runs)} runs)", xlabel="Epoch", ylabel="Loss")
    ax2 = row[1]
    for r in runs:
        ax2.plot(r.epoch, r[vcol], color=VAL, lw=0.6, alpha=0.18)
    ax2.plot(*mean_curve(runs, vcol), color=VAL, lw=2, label=vlabel)
    ax2.set(title=f"{title}: validation score", xlabel="Epoch", ylabel=vlabel.replace("Validation ", ""))
    for a in row:
        a.axvline(best, color=MUTED, lw=1, ls=":")
        a.text(best, 1.0, f" median best epoch {best}", transform=a.get_xaxis_transform(), color=MUTED, fontsize=7.5, va="top")
        a.grid(axis="y", color="#e6e5df", lw=0.6)
        a.legend(frameon=False, fontsize=7.5, loc="lower left" if final and a is row[0] else "best")
        a.title.set_fontsize(9.5)
fig.suptitle("Training and validation curves, 5-fold cross-validation (thin: individual runs; thick: mean)", fontsize=11, color=INK)
OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)
print(f"wrote {OUT}.png")
