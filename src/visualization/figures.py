"""Publication figures 2-8, drawn only from saved experimental outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# reference categorical palette, fixed order; colour follows the model, never its rank
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["B1 IMG (resnet50)", "B2 IMG (efficientnet_b0)", "B3 IMG (densenet121)", "A0", "B4", "A1", "A6", "M6"]
MODEL_LABEL = {"B1 IMG (resnet50)": "B1 Image-level ResNet50", "B2 IMG (efficientnet_b0)": "B2 Image-level EffNet-B0",
               "B3 IMG (densenet121)": "B3 Image-level DenseNet121", "A0": "A0 Mean-pool MIL", "B4": "B4 Attention MIL",
               "A1": "A1/B5 Task-aware MIL", "A2": "A2/B6 Structured MIL", "A3": "A3 + view dropout",
               "A4": "A4 + VD + consistency", "A5": "A5/B7 Structured + VD", "A6": "A6/B8 CoRe-MIL", "M2": "M2 Mean + structure*",
               "M3": "M3 Mean + VD*", "M5": "M5 Mean + structure + VD*", "M6": "M6 CoRe-MIL, mean pooling*"}
COLOR = {m: PALETTE[i] for i, m in enumerate(MODEL_ORDER)}
TASKS = [("hp", "H. pylori"), ("atrophy", "Atrophy presence"), ("distribution", "Atrophy distribution"),
         ("type", "Gastritis type")]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_dataset(audit: Path, out: Path):
    cls = pd.read_csv(audit / "class_distribution.csv")
    pp = pd.read_csv(audit / "patient_distribution.csv")
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1.6, 1.3, 1.4]))
    for ax, (t, name) in zip(axes, TASKS):
        d = cls[cls.task == t]
        x = np.arange(len(d))
        ax.bar(x, d.patients_metadata, width=0.6, color=PALETTE[0])
        for xi, (p, im) in enumerate(zip(d.patients_metadata, d.images_metadata)):
            ax.text(xi, p + 2, f"{p}\n({im} img)", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [c.replace("_", " ") for c in d.class_value], rotation=30, ha="right", fontsize=7.5)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_ylim(0, d.patients_metadata.max() * 1.35)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Patients")
    ax = axes[4]
    ax.hist(pp.n_images, bins=range(0, 112, 4), color=PALETTE[0], edgecolor="white", linewidth=1)
    ax.set_xlabel("Images per patient")
    ax.set_title(f"Bag size (median {pp.n_images.median():.0f})", fontsize=9)
    save(fig, out, "fig2_dataset")


def _models(df, prefer=MODEL_ORDER):
    return [m for m in prefer if m in set(df.model)]


def fig_baselines(long: pd.DataFrame, out: Path, scheme: str):
    full = long[(long.frac == 1.0) & (long.cal == "cal")]
    models = _models(full)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=False)
    w = 0.8 / len(models)
    for ax, metric, title in zip(axes, ["macro_f1", "balanced_accuracy"], ["Macro-F1", "Balanced accuracy"]):
        for i, m in enumerate(models):
            vals, lo, hi = [], [], []
            for t, _ in TASKS:
                r = full[(full.model == m) & (full.task == t) & (full.metric == metric)].iloc[0]
                vals.append(r.estimate)
                lo.append(r.estimate - r.get("lo", r.estimate))
                hi.append(r.get("hi", r.estimate) - r.estimate)
            x = np.arange(len(TASKS)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, vals, width=w * 0.9, color=COLOR[m], label=MODEL_LABEL[m])
            ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
        ax.set_xticks(range(len(TASKS)), [n.replace(" ", "\n", 1) for _, n in TASKS])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} ({'pooled 5-fold CV, n=195' if scheme == 'cv' else 'test cohort, n=34'})", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    save(fig, out, "fig3_baselines")


def fig_confusion(conf: dict, classes: list[str], out: Path):
    models = [m for m in ["B3 IMG (densenet121)", "A0", "A1", "A6", "M6"] if m in conf]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.3))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = np.array(conf[m])
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if rn[i, j] > 0.55 else INK)
        short = [c.replace("_predominant", "").replace("antrum_and_corpus", "both").replace("_", " ") for c in classes]
        ax.set_xticks(range(len(classes)), short, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), short)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(MODEL_LABEL.get(m, m), fontsize=9)
    axes[0].set_ylabel("True")
    fig.suptitle("Atrophy distribution: seed-averaged patient counts (colour = row-normalised)", fontsize=9, color=INK2)
    save(fig, out, "fig4_confusion_distribution")


def fig_missing_view(long: pd.DataFrame, out: Path):
    d = long[(long.cal == "cal") & (long.task.isin(["distribution", "mean"])) & (long.metric.isin(["macro_f1", "ece"]))]
    models = [m for m in ["B2 IMG (efficientnet_b0)", "B3 IMG (densenet121)", "A0", "A1", "A6", "M6"] if m in set(d.model)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (task, metric, title) in zip(axes, [("distribution", "macro_f1", "Atrophy distribution macro-F1"),
                                                ("mean", "macro_f1", "Mean macro-F1 (4 tasks)"),
                                                ("distribution", "ece", "Atrophy distribution ECE")]):
        for m in models:
            s = d[(d.model == m) & (d.task == task) & (d.metric == metric)].sort_values("frac")
            sd = np.sqrt(s.rep_sd ** 2 + s.seed_sd ** 2)
            ax.errorbar(s.frac * 100, s.estimate, yerr=sd, color=COLOR[m], marker="o", ms=4.5, lw=2, capsize=0,
                        label=MODEL_LABEL[m])
        ax.set_xticks([25, 50, 75, 100])
        ax.set_xlabel("Images available per patient (%)")
        ax.set_title(title, fontsize=9)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    fig.text(0.01, -0.04, "Random image subsets (no anatomical labels). Error bars: sqrt(subset SD² + seed SD²).",
             fontsize=7.5, color=INK2)
    save(fig, out, "fig5_missing_view")


def fig_ablation(long: pd.DataFrame, out: Path):
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "M2", "M3", "M5", "M6"]
    full = long[(long.frac == 1.0) & (long.cal == "cal") & (long.task == "distribution")]
    q = long[(long.frac == 0.25) & (long.cal == "cal") & (long.task == "distribution") & (long.metric == "macro_f1")]
    order = [m for m in order if m in set(full.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (metric, title) in zip(axes, [("macro_f1", "Atrophy distribution macro-F1 (100% images)"),
                                          ("balanced_accuracy", "Atrophy distribution balanced accuracy")]):
        for i, m in enumerate(order):
            r = full[(full.model == m) & (full.metric == metric)].iloc[0]
            ax.plot([r.lo, r.hi], [i, i], color=INK2, lw=1.2)
            ax.plot(r.estimate, i, "o", ms=7, color=PALETTE[0], mec="white", mew=1.5)
            if metric == "macro_f1" and len(q[q.model == m]):
                ax.plot(q[q.model == m].estimate.iloc[0], i, "D", ms=5.5, color=PALETTE[1], mec="white", mew=1)
        ax.set_title(title, fontsize=9)
        n_pre = sum(not m.startswith("M") for m in order)
        if n_pre < len(order):
            ax.axhline(n_pre - 0.5, color=MUTED, lw=0.8, ls=":")
    axes[0].set_yticks(range(len(order)), [MODEL_LABEL[m] for m in order])
    axes[0].invert_yaxis()  # shared y: invert once
    axes[0].plot([], [], "o", color=PALETTE[0], label="100% images (95% patient-bootstrap CI)")
    axes[0].plot([], [], "D", color=PALETTE[1], label="25% images (point estimate)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.09, "* post-hoc exploratory variants (below dotted line), defined after the pre-specified CV results.",
             fontsize=7.5, color=INK2)
    save(fig, out, "fig6_ablation")


def fig_calibration(rel: pd.DataFrame, long: pd.DataFrame, out: Path):
    models = [m for m in ["B2 IMG (efficientnet_b0)", "A0", "A1", "A6", "M6"] if m in set(rel.model)]
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.0 * (len(models) + 1), 3.1))
    for ax, m in zip(axes, models):
        for cal, mk, col in (("raw", "s", MUTED), ("cal", "o", COLOR[m])):
            r = rel[(rel.model == m) & (rel.cal == cal) & (rel.n > 0)]
            ax.plot(r.conf, r.acc, marker=mk, ms=5, lw=1.8, color=col, label="before" if cal == "raw" else "temperature-scaled")
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
        ax.set_xlim(0.2, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(MODEL_LABEL[m], fontsize=8.5)
    axes[0].set_ylabel("Accuracy (distribution)")
    axes[0].legend(fontsize=7, loc="upper left")
    ax = axes[-1]
    f = long[(long.frac == 1.0) & (long.task == "distribution") & (long.metric == "ece")]
    for i, m in enumerate(models):
        for j, cal in enumerate(("raw", "cal")):
            v = f[(f.model == m) & (f.cal == cal)].estimate.iloc[0]
            ax.bar(i + (j - 0.5) * 0.38, v, width=0.34, color=COLOR[m] if cal == "cal" else GRID,
                   edgecolor=COLOR[m], linewidth=1)
    ax.set_xticks(range(len(models)), [MODEL_LABEL[m].split(" ")[0].split("/")[0] for m in models])
    ax.set_title("ECE: before (light) / after", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig7_calibration")


def fig_attention(attn: pd.DataFrame, preds: pd.DataFrame, images: np.ndarray, index: pd.DataFrame, out: Path,
                  classes: dict, k: int = 4, n_patients: int = 3):
    """Top-k attended images per task branch for a few held-out patients (correct and incorrect)."""
    pos = {im: i for i, im in enumerate(index.image_id)}
    d = preds[(preds.frac == 1.0) & (preds.cal == "cal") & (preds.task == "distribution")].copy()
    pcols = [c for c in d.columns if c.startswith("p") and c[1:].isdigit()]
    d["pred"] = d[pcols].to_numpy().argmax(1)
    d["correct"] = d.pred == d.label
    chosen = list(d[d.correct].sort_values("n_images").patient_id.iloc[[len(d[d.correct]) // 2]]) + \
        list(d[~d.correct].patient_id.iloc[:max(0, n_patients - 1)])
    branches = [b for b in ["hp", "distribution"] if b in set(attn.branch)]
    fig, axes = plt.subplots(len(chosen) * len(branches), k, figsize=(1.9 * k, 2.0 * len(chosen) * len(branches)))
    axes = np.atleast_2d(axes)
    row = 0
    for p in chosen:
        info = d[d.patient_id == p].iloc[0]
        for b in branches:
            a = attn[(attn.patient_id == p) & (attn.branch == b)].sort_values("weight", ascending=False)
            for j in range(k):
                ax = axes[row, j]
                ax.axis("off")
                if j < len(a):
                    r = a.iloc[j]
                    ax.imshow(images[pos[r.image_id]])
                    ax.set_title(f"w={r.weight:.2f}", fontsize=7, color=INK2)
            axes[row, 0].text(-0.08, 0.5, f"case {chosen.index(p) + 1}\n{b} branch\n"
                              f"true: {classes[int(info.label)]}\npred: {classes[int(info.pred)]}",
                              transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=7, color=INK)
            row += 1
    fig.suptitle("Highest-weighted images per task-specific attention branch (held-out patients).\nAttention weights "
                 "indicate model reliance, not lesion localisation.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.12, 0, 1, 0.97))
    save(fig, out, "fig8_attention")


def attention_stats(attn: pd.DataFrame) -> pd.DataFrame:
    """Per patient: entropy of each branch and rank correlation between branches."""
    from scipy.stats import spearmanr
    rows = []
    for p, g in attn.groupby("patient_id"):
        piv = g.pivot_table(index="image_id", columns="branch", values="weight")
        r = dict(patient_id=p, n_images=len(piv))
        for b in piv.columns:
            w = piv[b].to_numpy()
            r[f"norm_entropy_{b}"] = float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w))) if len(w) > 1 else np.nan
            r[f"max_weight_{b}"] = float(w.max())
        if {"hp", "distribution"} <= set(piv.columns) and len(piv) > 2:
            r["spearman_hp_vs_distribution"] = float(spearmanr(piv["hp"], piv["distribution"]).statistic)
        rows.append(r)
    return pd.DataFrame(rows)
