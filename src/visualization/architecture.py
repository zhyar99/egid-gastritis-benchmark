"""Figure 1: CoRe-MIL architecture schematic (drawn, contains no results)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, BLUE, ORANGE, AQUA, VIOLET = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def _box(ax, x, y, w, h, text, color, fs=8.5, fill=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", lw=1.2,
                                ec=color, fc=color + f"{int(fill * 255):02x}"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def _arrow(ax, a, b, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, lw=1.1, color=color, linestyle=ls))


def draw(path_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 0.5)
    ax.axis("off")
    # images
    for i in range(4):
        _box(ax, 0.01 + i * 0.008, 0.30 - i * 0.03, 0.07, 0.07, "" if i < 3 else "image\n$x_{i,k}$", MUTED, 7, 0.05)
    ax.text(0.05, 0.40, "Patient bag\n$X_i=\\{x_{i,1..N_i}\\}$", ha="center", fontsize=8.5)
    _box(ax, 0.13, 0.22, 0.11, 0.12, "Shared encoder\nEfficientNet-B0\n(frozen or fine-tuned)", BLUE)
    _arrow(ax, (0.105, 0.28), (0.13, 0.28))
    _box(ax, 0.27, 0.22, 0.09, 0.12, "Projection\n$h_{i,k}$ = MLP$(f_{i,k})$", BLUE)
    _arrow(ax, (0.24, 0.28), (0.27, 0.28))
    # view dropout
    _box(ax, 0.27, 0.04, 0.20, 0.10, "Module E: view sampling\nteacher view $V^T$ (full bag)\nstudent view $V^S\\subset V^T$, |$V^S$|=r·N, r~U[0.2,0.8]",
         ORANGE, 7.5)
    _arrow(ax, (0.315, 0.22), (0.33, 0.14), ORANGE, ls="--")
    # attention branches
    _box(ax, 0.40, 0.33, 0.14, 0.09, "Gated attention $a^{(HP)}$\n(H. pylori branch)", AQUA, 8)
    _box(ax, 0.40, 0.19, 0.14, 0.09, "Gated attention $a^{(D)}$\n(distribution branch)", AQUA, 8)
    ax.text(0.47, 0.445, "Module C: task-aware masked attention", ha="center", fontsize=8, color=MUTED)
    _arrow(ax, (0.36, 0.30), (0.40, 0.37))
    _arrow(ax, (0.36, 0.26), (0.40, 0.235))
    # heads
    _box(ax, 0.58, 0.33, 0.12, 0.09, "$z^{HP}_i$ → softmax\n$p(\\mathrm{HP})$ (2 classes)", VIOLET, 8)
    _box(ax, 0.58, 0.19, 0.12, 0.09, "$z^{D}_i$ → softmax\n$p(\\mathrm{D})$ (4 classes)", VIOLET, 8)
    _arrow(ax, (0.54, 0.375), (0.58, 0.375))
    _arrow(ax, (0.54, 0.235), (0.58, 0.235))
    # structured
    _box(ax, 0.725, 0.19, 0.185, 0.13, "Module D: structured head\n$p(\\mathrm{atrophy}) = p(\\mathrm{D})\\,M_A$\n$p(\\mathrm{type}) = p(\\mathrm{D})\\,M_T$\n($M$ re-derived on training patients)",
         VIOLET, 7.5)
    _arrow(ax, (0.70, 0.235), (0.725, 0.235))
    _box(ax, 0.94, 0.25, 0.15, 0.17, "Module G: outputs\n4 patient-level tasks\nclass probabilities\nconfidence, entropy\ntemperature-scaled\n(validation patients)", MUTED, 7.5)
    _arrow(ax, (0.70, 0.375), (0.94, 0.375))
    _arrow(ax, (0.91, 0.27), (0.94, 0.30))
    # consistency
    _box(ax, 0.58, 0.03, 0.32, 0.11, "Module F: confidence-gated cross-view consistency\n"
         "$L_{cv}=T^2\\,\\mathbb{1}[\\max p^T\\geq\\tau]\\;\\mathrm{KL}(\\mathrm{sg}[p^T_{/T}]\\;\\|\\;p^S_{/T})$, warm-up ramp\n"
         "$L = L_{sup}(V^T)/2 + L_{sup}(V^S)/2 + \\lambda_2 L_{cv}$", ORANGE, 7.5)
    _arrow(ax, (0.47, 0.09), (0.58, 0.09), ORANGE, ls="--")
    _arrow(ax, (0.66, 0.19), (0.66, 0.14), ORANGE, ls="--")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
