"""Predict the four gastritis labels for ONE patient from a folder of their endoscopic images
(works with any model file written by scripts/train_final.py: CoRe-MIL or the image-level CNN).

  python scripts/predict.py path/to/patient_folder
  python scripts/predict.py path/to/patient_folder --json result.json
  python scripts/predict.py path/to/patient_folder --model models/cnn_densenet121_final.pt

Research use only: not a medical device, not clinically validated (single-centre data, no external
validation). Probabilities are temperature-scaled model scores, not clinical certainty.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402
from src.data.metadata import IMAGE_EXTS  # noqa: E402
from src.data.preprocessing import load_square  # noqa: E402
from src.models.baseline import ImageLevelModel  # noqa: E402
from src.models.core_mil import MILModel  # noqa: E402
from src.models.encoders import Normalize  # noqa: E402
from src.models.structured_head import coherence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = dict(hp="H. pylori infection", atrophy="Presence of atrophy", distribution="Distribution of atrophy",
                  type="Gastritis type")


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "coremil-final-v1":
        raise ValueError(f"{path} is not a CoRe-MIL final model file")
    # encoder build options (e.g. DINOv2 img_size); files written before they were stored fall back to the config
    kw = ck.get("encoder_kwargs") or load_config()["encoders"].get(ck["encoder"], {}).get("kwargs", {})
    enc = timm.create_model(ck["encoder_timm"], pretrained=False, num_classes=0, **kw)
    enc.load_state_dict(ck["encoder_state"])
    v = ck["variant_config"]
    if ck.get("kind", "mil") == "image":
        head = ImageLevelModel(enc.num_features, ck["n_classes"], hidden=v["hidden"], dropout=v["dropout"],
                               input_dropout=v.get("input_dropout", 0.0), in_norm=ck.get("in_norm", False))
        head.load_state_dict(ck["head_state"])
        return ck, enc.eval(), head.eval()
    head = MILModel(enc.num_features, ck["n_classes"], pooling=v["pooling"], head=ck["head"],
                    mapping={k: np.array(m, np.float32) for k, m in ck["mapping"].items()}, hidden=v["hidden"],
                    attn_dim=v["attn_dim"], dropout=v["dropout"], input_dropout=v.get("input_dropout", 0.0),
                    in_norm=ck.get("in_norm", False),
                    **{k: v[k] for k in ("topk", "mask_topk", "mask_p", "tr_layers", "tr_heads", "tr_ffn") if k in v})
    head.load_state_dict(ck["head_state"])
    return ck, enc.eval(), head.eval()


@torch.no_grad()
def predict(folder: Path, model_path: Path, device: str = "auto", top_k: int = 3) -> dict:
    ck, enc, head = load_model(model_path)
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) in {folder}")
    imgs, used, skipped = [], [], []
    # fine-tuned models were evaluated on the 256-px cache resized to 224; frozen ones on a direct 224 resize
    load_size = ck.get("resize_from") or ck["image_size"]
    for p in files:
        try:
            imgs.append(load_square(p, load_size))
            used.append(p.name)
        except Exception as e:  # unreadable image: skip it, report it
            skipped.append(f"{p.name}: {e}")
    if not imgs:
        raise ValueError(f"none of the {len(files)} images could be read")
    dev = torch.device("mps" if device == "auto" and torch.backends.mps.is_available() else
                       ("cpu" if device == "auto" else device))
    enc, norm = enc.to(dev), Normalize(ck["norm_mean"], ck["norm_std"]).to(dev)
    x = torch.from_numpy(np.stack(imgs))
    if ck.get("resize_from"):
        from src.training.finetune import transforms
        x = transforms(ck["image_size"], train=False)(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
    feats = torch.cat([enc(norm(x[s: s + 32].to(dev))).cpu() for s in range(0, len(x), 32)])
    feats = (feats - ck["feat_mean"]) / ck["feat_std"]
    image_level = ck.get("kind", "mil") == "image"
    if image_level:  # per-image prediction, patient = mean of image probabilities (EGID protocol)
        img_logits = head(feats)
        out = dict(logits=ImageLevelModel.aggregate(img_logits, torch.zeros(len(feats), dtype=torch.long), 1),
                   attn=None)
    else:
        out = head(feats[None], torch.ones(1, len(feats), dtype=torch.bool))
    logp = head.log_probs(out, ck["temperatures"])
    res = dict(folder=str(folder), n_images=len(used), skipped=skipped, model=str(model_path),
               disclaimer="Research use only; not clinically validated.", tasks={})
    pred_idx = {}
    for t, lp in logp.items():
        p = lp.exp()[0].numpy()
        k = int(p.argmax())
        pred_idx[t] = np.array([k])
        ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(len(p)))
        res["tasks"][t] = dict(name=TASK_NAMES[t], prediction=ck["classes"][t][k], confidence=float(p[k]),
                               normalized_entropy=ent,
                               probabilities={c: float(q) for c, q in zip(ck["classes"][t], p)},
                               derived_from_distribution=ck["head"] == "structured" and t in ck["mapping"])
    res["coherent"] = bool(coherence(pred_idx, {k: np.array(m) for k, m in ck["mapping"].items()})[0])
    if image_level:  # images whose own prediction most supports the patient-level distribution class
        pimg = img_logits["distribution"].softmax(-1)[:, pred_idx["distribution"][0]]
        w, i = torch.topk(pimg, min(top_k, len(used)))
        res["top_supporting_images"] = {"distribution": [dict(image=used[j], p=float(q)) for q, j in zip(w, i)]}
    if out["attn"] is not None:
        res["top_attended_images"] = {g: [dict(image=used[i], weight=float(w)) for w, i in
                                          zip(*torch.topk(out["attn"][0, b], min(top_k, len(used))))]
                                      for b, g in enumerate(head.groups if out["attn"].shape[1] > 1 else ["shared"])}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="folder with one patient's images")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "coremil_final.pt")
    ap.add_argument("--json", type=Path, help="also write the result as JSON")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    a = ap.parse_args()
    r = predict(a.folder, a.model, a.device)
    print(f"\nPatient folder: {r['folder']}  ({r['n_images']} images" +
          (f", {len(r['skipped'])} unreadable skipped" if r["skipped"] else "") + ")")
    for t in r["tasks"].values():
        probs = ", ".join(f"{c} {q:.2f}" for c, q in t["probabilities"].items())
        print(f"  {t['name']:<24} {t['prediction']:<20} conf {t['confidence']:.2f}   [{probs}]"
              + ("   (derived)" if t["derived_from_distribution"] else ""))
    print(f"  Label combination clinically consistent: {r['coherent']}")
    for g, lst in r.get("top_attended_images", {}).items():
        print(f"  Most relied-on images ({g} branch): " + ", ".join(f"{d['image']} ({d['weight']:.2f})" for d in lst))
    for g, lst in r.get("top_supporting_images", {}).items():
        print(f"  Images most supporting the {g} prediction: " + ", ".join(f"{d['image']} (p={d['p']:.2f})" for d in lst))
    print(f"  {r['disclaimer']}")
    if a.json:
        a.json.write_text(json.dumps(r, indent=2))
        print(f"  saved {a.json}")
