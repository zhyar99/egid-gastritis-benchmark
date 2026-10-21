"""Supplementary: Grad-CAM for the end-to-end fine-tuned CoRe-MIL (FT_A6) on held-out CV patients.

The gradient of the patient-level atrophy-distribution logit (predicted class) is taken w.r.t. the
last convolutional feature map of every image in the bag, so each map shows which regions of an image
raised the *patient* score. Shown for the images with the highest distribution-branch attention.
Grad-CAM is coarse (7x7 for EfficientNet-B0 at 224 px), not validated against lesions and not proof
of pathology localisation.

  python scripts/gradcam.py --split cv0
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402
from src.data.dataset import Cohort, pad_bags  # noqa: E402
from src.data.preprocessing import open_cache  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.models.core_mil import MILModel  # noqa: E402
from src.models.encoders import Normalize, create_encoder  # noqa: E402
from src.visualization.figures import INK, INK2, save, style  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    cfg["variants"].update(cfg["finetune_variants"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="cv0")
    ap.add_argument("--encoder", default="efficientnet_b0")
    ap.add_argument("--k", type=int, default=4, help="images per patient")
    a = ap.parse_args()
    P = cfg["paths"]
    run_id = f"FT_A6__{a.encoder}_224__{a.split}__s0"
    co = Cohort.load(P["data"], P["audit"])
    ck = torch.load(P["checkpoints"] / f"{run_id}.pt", weights_only=False)
    v = ck["variant"]
    enc, info = create_encoder(a.encoder, cfg["encoders"], pretrained=False)
    enc.load_state_dict(torch.load(P["checkpoints"] / f"{run_id}_encoder.pt", weights_only=True))
    head = MILModel(info["dim"], co.n_classes, pooling=v["pooling"], head=ck["head"], mapping=ck["mapping"],
                    hidden=v["hidden"], attn_dim=v["attn_dim"], dropout=v["dropout"], in_norm=True)
    head.load_state_dict(ck["state"])
    enc.eval(), head.eval()
    norm = Normalize(info["mean"], info["std"])
    images = open_cache(P["cache"], cfg["finetune"]["cache_size"], co.index)
    dist = next(t for t in co.tasks if t.key == "distribution")

    preds = pd.read_csv(P["outputs"] / "predictions" / f"{run_id}.csv.gz", dtype={"patient_id": str})
    d = preds[(preds.set == "test") & (preds.frac == 1.0) & (preds.cal == "cal") & (preds.task == "distribution")].copy()
    d["pred"] = d[[f"p{c}" for c in range(dist.n_classes)]].to_numpy().argmax(1)
    test_ids = load_split(P["splits"], a.split)["test"]
    assert set(d.patient_id) == set(test_ids)
    correct = d[d.pred == d.label].sort_values("n_images")
    wrong = d[d.pred != d.label].sort_values("n_images")
    chosen = [correct.patient_id.iloc[len(correct) // 2]] + list(wrong.patient_id.iloc[:1])

    style()
    fig, axes = plt.subplots(len(chosen), a.k, figsize=(2.1 * a.k, 2.4 * len(chosen)))
    axes = np.atleast_2d(axes)
    for r, pid in enumerate(chosen):
        rows = co.bags[pid]
        raw = torch.from_numpy(np.ascontiguousarray(images[rows])).permute(0, 3, 1, 2)
        x = norm(F.interpolate(raw.float(), size=(224, 224), mode="bilinear", antialias=True)
                 .clamp(0, 255).permute(0, 2, 3, 1))
        fmap = enc.forward_features(x)  # [N, C, 7, 7]
        fmap.retain_grad()
        feats = enc.forward_head(fmap)
        out = head(*pad_bags(feats, [np.arange(len(rows))]))
        logits = out["logits"]["distribution"][0]
        cls = int(logits.argmax())
        logits[cls].backward()
        w = fmap.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * fmap).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0].detach().numpy()
        cam /= cam.reshape(len(cam), -1).max(1)[:, None, None] + 1e-8
        att = out["attn"][0, head.groups.index("distribution")].detach().numpy()
        top = np.argsort(-att)[: a.k]
        lab = int(d[d.patient_id == pid].label.iloc[0])
        for j, i in enumerate(top):
            ax = axes[r, j]
            ax.imshow(F.interpolate(raw[i: i + 1].float(), size=(224, 224), mode="bilinear",
                                    antialias=True)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy())
            ax.imshow(cam[i], cmap="magma", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"attention {att[i]:.2f}", fontsize=7, color=INK2)
            ax.axis("off")
        axes[r, 0].text(-0.08, 0.5, f"{'correct' if cls == lab else 'error'}\ntrue: {dist.classes[lab]}\n"
                        f"pred: {dist.classes[cls]}", transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=7, color=INK)
    fig.suptitle(f"Grad-CAM of the patient-level distribution logit, end-to-end CoRe-MIL ({a.split} held-out)."
                 "\nCoarse 7×7 maps; not validated against lesions.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.12, 0, 1, 0.95))
    save(fig, P["outputs"] / "figures" / "cv", "figS_gradcam_ft_a6")
    print("chosen", chosen)
