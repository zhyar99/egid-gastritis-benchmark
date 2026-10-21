# Model card: CoRe-MIL final (`coremil_final.pt`)

**Research use only. Not a medical device, not clinically validated.**

## What it is

The pre-specified full CoRe-MIL (variant A6) trained once on all 229 EGID patients (5,883 images).
Input: all endoscopic images of one patient (any number). Output, per patient:

| Task | Classes | How it is produced |
|---|---|---|
| H. pylori infection | positive, negative | own attention branch + head |
| Distribution of atrophy | normal, antrum-predominant, corpus-predominant, antrum-and-corpus | own attention branch + head |
| Presence of atrophy | atrophic, non-atrophic | **derived** from the distribution probabilities |
| Gastritis type | type A, type B, normal | **derived** from the distribution probabilities |

Plus confidence, normalised entropy, a consistency flag for the four decisions, and the images each
attention branch relied on most. How the model works: `research/methodology.md`.

## Components

- Encoder: EfficientNet-B0 (`timm` `efficientnet_b0.ra_in1k`, ImageNet weights, **frozen**, stored in the file).
- Preprocessing: RGB → black padding to square → bicubic resize to 224 px → ImageNet mean/std;
  features z-scored with the training-image statistics stored in the file.
- Head: MLP (256) → task-specific gated attention → H. pylori and distribution heads → structured mapping.
- Training: view sampling (full bag + 20–80% subset) and confidence-gated cross-view consistency.

## How it was trained

- All 229 patients, seed 0, 15 epochs, AdamW lr 3e-4, wd 1e-2, dropout 0.5, input dropout 0.3.
- No patient was held out, so **this model has no evaluation of its own**. The two settings that
  normally need validation data were fixed from the 25 cross-validation runs of the identical
  configuration: epochs = median best epoch + 1 (= 15); temperatures = median fitted temperature
  (H. pylori 0.67, distribution 0.92).
- Rebuild: `python scripts/train_final.py` (refuses to overwrite an existing model file).

## Expected performance (from the same configuration, not from this file)

| Evaluation | Distribution macro-F1 | H. pylori macro-F1 | Mean macro-F1 (4 tasks) |
|---|---|---|---|
| 5-fold patient-level CV, 195 patients, 5 seeds | 0.580 [0.518, 0.638] | 0.865 | 0.699 |
| Untouched test cohort, 34 patients, 5 seeds | 0.575 [0.416, 0.700] | 0.953 | 0.716 |

Decisions were clinically consistent for ~90% of CV patients. Expect lower performance with fewer
images (CV distribution macro-F1 0.545 with 25% of images).

**In the same experiments simpler models performed better**: image-level probability averaging with
EfficientNet-B0 or DenseNet121 reached 0.722 CV distribution macro-F1, mean-pooling MIL 0.677 and
the post-hoc mean-pooling CoRe-MIL (M6) 0.686. If predictive accuracy matters more than using the
proposed architecture, export M6 instead (`scripts/train_final.py --variant M6 --out
models/m6_final.pt`); the image-level model would need a small separate exporter (not written).

## Use

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --json result.json
```

## Limitations

Single centre (Renji Hospital), single endoscope (Olympus CV-290, white light), curated high-quality
images, cardia/pylorus excluded, 229 patients (31 corpus-predominant). No external or prospective
validation. Attention weights show what the model relied on, not where pathology is. Presence of
atrophy and gastritis type are derived from the distribution prediction under EGID's annotation
scheme and are not independent findings. Full list: `research/limitations.md`.

---

# Model card: image-level CNN final (`cnn_densenet121_final.pt`)

**Research use only. Not a medical device, not clinically validated.**

- **What it is:** the strongest CNN of the study: DenseNet121 (`timm` `densenet121.tv_in1k`,
  ImageNet weights, frozen, stored in the file) + a trained MLP (256) with four independent task heads,
  applied to every image; the patient prediction is the **mean of the image probabilities** (EGID
  protocol). Atrophy presence and gastritis type have their own heads (not derived), so decisions can be
  inconsistent; `predict.py` reports a consistency flag.
- **Training:** all 229 patients, seed 0, 53 epochs (= median best epoch + 1 of its 25 CV runs),
  AdamW lr 3e-4, wd 1e-2, dropout 0.25, input dropout 0.3; temperatures = medians of the CV fits
  (H. pylori 0.60, presence 1.15, distribution 0.80, type 0.90). No held-out evaluation of this file.
- **Expected performance (same configuration):**

| Evaluation | Mean accuracy (4 tasks) | H. pylori acc | Presence acc | Distribution acc | Type acc | Distribution macro-F1 |
|---|---|---|---|---|---|---|
| 5-fold CV, 195 patients, 5 seeds | 79.5% | 88.5% | 82.4% | 68.9% [62.9, 75.2] | 78.4% | 0.722 [0.663, 0.778] |
| Test cohort, 34 patients, 5 seeds | 75.0% | 100% | 71.2% | 65.3% | 63.5% | 0.669 [0.455, 0.816] |

  Decisions were consistent for ~90% of CV patients.
- **Use:**

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --model models/cnn_densenet121_final.pt
```

- **Rebuild:** `python scripts/train_final.py --variant IMG --encoder densenet121 --out models/cnn_densenet121_final.pt`
- Limitations as above. Fine-tuned (all layers trainable) CNNs were evaluated under protocol Addendum A
  (`research/experiment_protocol.md` §10, 2 seeds): none differed from its frozen counterpart (fine-tuned
  − frozen distribution macro-F1: DenseNet121 −0.019 [−0.073, 0.032], EfficientNet-B0 +0.008 [−0.041, 0.056],
  ResNet50 −0.024 [−0.083, 0.033]), so this frozen model is kept.

---

# Model card: fine-tuned CoRe-MIL final (`coremil_ft_final.pt`)

**Research use only. Not a medical device, not clinically validated.**

- **What it is:** CoRe-MIL (A6 architecture) trained **end to end** (variant FT_A6): the EfficientNet-B0
  encoder (`timm` `efficientnet_b0.ra_in1k`, ImageNet start) is trained together with the MIL head, both
  stored in the file. Same outputs as `coremil_final.pt` (atrophy presence and gastritis type derived).
- **Preprocessing:** RGB → black padding to square → 256 px cache → antialiased bilinear 224 px → ImageNet
  mean/std; a LayerNorm on the embeddings replaces the frozen-feature z-scoring. `predict.py` does this.
- **Training:** all 229 patients (5,883 images), seed 0, 20 epochs (= median best epoch + 1 of its 15 CV
  runs), AdamW lr 3e-4, wd 1e-2, dropout 0.25, 12 images per patient per step, MPS GPU, 19.5 min;
  temperatures = medians of the CV fits (H. pylori 0.69, distribution 0.75). No held-out evaluation of this file.
- **Expected performance (same configuration, 3 seeds):**

| Evaluation | Mean accuracy (4 tasks) | H. pylori acc | Presence acc | Distribution acc | Type acc | Distribution macro-F1 |
|---|---|---|---|---|---|---|
| 5-fold CV, 195 patients | 71.8% | 81.0% | 76.9% | 60.2% | 68.9% | 0.617 [0.558, 0.671] |
| Test cohort, 34 patients | 73.8% | 98.0% | 75.5% | 53.9% | 67.6% | 0.563 [0.422, 0.675] |

  Decisions were consistent for 92.8% of CV patients. With 25% of the images: CV distribution macro-F1 0.547.
- **Compared with the other exported models:** better than frozen CoRe-MIL in CV (+0.037 [−0.030, 0.101],
  not significant) and **worse than the image-level DenseNet121** (−0.105 [−0.166, −0.045]).
- **Use:**

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --model models/coremil_ft_final.pt
```

- **Rebuild:** `python scripts/train_final.py --variant FT_A6 --out models/coremil_ft_final.pt`
- Limitations as above.

---

# All final models (`models/final/`)

**Research use only. Not a medical device, not clinically validated.**

Every configuration with cross-validation runs was also trained once on all 229 patients with
`scripts/train_final.py` (epochs = median best CV epoch + 1, temperatures = median CV fits). No patient
was held out, so **none of these files has an evaluation of its own**: the numbers below are the pooled
5-fold CV results (195 patients) of the identical configuration. They are sorted by distribution macro-F1.
Encoder-sweep rows (non-EfficientNet-B0 MIL models and image-level ConvNeXt/Swin/DINOv2) used 3 seeds and
only their macro-F1 is pooled (–: not computed); fine-tuned rows used 1–3 seeds; all others 5 seeds.
C1 is exploratory (secondary comparison of Addendum C). Variant definitions: `configs/experiments.yaml`.

Use any file the same way:

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --model models/final/C1_efficientnet_b0.pt
```

| File | Variant | Encoder | Epochs | Mean acc. (%) | H. pylori acc. (%) | Distribution macro-F1 [95% CI] | Coherent (%) |
|---|---|---|---|---|---|---|---|
| `models/final/FT2_IMG_efficientnet_b0.pt` | FT2_IMG | efficientnet_b0 | 8 | 78.3 | 86.9 | 0.730 [0.671, 0.782] | 84.6 |
| `models/final/IMG_swin_tiny.pt` | IMG | swin_tiny | 118 | – | – | 0.727 [0.668, 0.779] | – |
| `models/final/IMG_efficientnet_b0.pt` | IMG | efficientnet_b0 | 93 | 76.1 | 87.5 | 0.722 [0.665, 0.773] | 79.0 |
| `models/cnn_densenet121_final.pt` | IMG | densenet121 | 53 | 79.5 | 88.5 | 0.722 [0.663, 0.778] | 89.8 |
| `models/final/A0_swin_tiny.pt` | A0 | swin_tiny | 57 | – | – | 0.705 [0.643, 0.764] | – |
| `models/final/FT2_IMG_densenet121.pt` | FT2_IMG | densenet121 | 12 | 77.4 | 88.7 | 0.703 [0.646, 0.756] | 85.6 |
| `models/final/IMG_convnext_tiny.pt` | IMG | convnext_tiny | 131 | – | – | 0.697 [0.632, 0.752] | – |
| `models/final/IMG_dinov2_small.pt` | IMG | dinov2_small | 144 | – | – | 0.693 [0.632, 0.748] | – |
| `models/final/A0_densenet121.pt` | A0 | densenet121 | 40 | – | – | 0.692 [0.627, 0.752] | – |
| `models/final/M5_efficientnet_b0.pt` | M5 | efficientnet_b0 | 57 | 77.9 | 90.1 | 0.687 [0.624, 0.742] | 94.5 |
| `models/final/M6_efficientnet_b0.pt` | M6 | efficientnet_b0 | 72 | 77.4 | 87.9 | 0.686 [0.621, 0.741] | 96.1 |
| `models/final/M3_efficientnet_b0.pt` | M3 | efficientnet_b0 | 58 | 78.1 | 90.5 | 0.684 [0.619, 0.739] | 89.6 |
| `models/final/A0_efficientnet_b0.pt` | A0 | efficientnet_b0 | 52 | 77.2 | 90.5 | 0.677 [0.613, 0.731] | 88.9 |
| `models/final/M2_efficientnet_b0.pt` | M2 | efficientnet_b0 | 59 | 76.6 | 89.0 | 0.676 [0.612, 0.731] | 94.7 |
| `models/final/IMG_resnet50.pt` | IMG | resnet50 | 77 | 74.6 | 87.6 | 0.672 [0.604, 0.734] | 83.9 |
| `models/final/A6_swin_tiny.pt` | A6 | swin_tiny | 40 | – | – | 0.668 [0.608, 0.723] | – |
| `models/final/A0_dinov2_small.pt` | A0 | dinov2_small | 95 | – | – | 0.667 [0.600, 0.726] | – |
| `models/final/FT2_IMG_resnet50.pt` | FT2_IMG | resnet50 | 9 | 75.2 | 89.0 | 0.648 [0.581, 0.706] | 84.6 |
| `models/final/A0_resnet50.pt` | A0 | resnet50 | 22 | – | – | 0.643 [0.577, 0.705] | – |
| `models/final/X3_efficientnet_b0.pt` | X3 | efficientnet_b0 | 68 | 74.4 | 87.9 | 0.641 [0.582, 0.694] | 93.7 |
| `models/final/C1_efficientnet_b0.pt` | C1 | efficientnet_b0 | 45 | 74.0 | 87.9 | 0.637 [0.579, 0.689] | 93.6 |
| `models/final/A6_densenet121.pt` | A6 | densenet121 | 29 | – | – | 0.632 [0.570, 0.691] | – |
| `models/final/A0_convnext_tiny.pt` | A0 | convnext_tiny | 59 | – | – | 0.628 [0.560, 0.689] | – |
| `models/final/A1_swin_tiny.pt` | A1 | swin_tiny | 16 | – | – | 0.627 [0.564, 0.685] | – |
| `models/final/N4_efficientnet_b0.pt` | N4 | efficientnet_b0 | 54 | 73.7 | 86.6 | 0.626 [0.568, 0.679] | 92.3 |
| `models/final/A1_densenet121.pt` | A1 | densenet121 | 10 | – | – | 0.625 [0.564, 0.684] | – |
| `models/coremil_ft_final.pt` | FT_A6 | efficientnet_b0 | 20 | 71.8 | 81.0 | 0.617 [0.558, 0.671] | 92.8 |
| `models/final/N1_efficientnet_b0.pt` | N1 | efficientnet_b0 | 46 | 73.2 | 88.8 | 0.616 [0.554, 0.672] | 79.1 |
| `models/final/FT_IMG_densenet121.pt` | FT_IMG | densenet121 | 7 | 74.9 | 86.2 | 0.614 [0.536, 0.685] | 82.1 |
| `models/final/N2_efficientnet_b0.pt` | N2 | efficientnet_b0 | 43 | 71.8 | 87.4 | 0.611 [0.551, 0.664] | 92.3 |
| `models/final/C3_efficientnet_b0.pt` | C3 | efficientnet_b0 | 62 | 72.0 | 85.6 | 0.607 [0.550, 0.658] | 92.8 |
| `models/final/A4_efficientnet_b0.pt` | A4 | efficientnet_b0 | 33 | 73.5 | 87.6 | 0.605 [0.543, 0.661] | 82.5 |
| `models/final/A6_resnet50.pt` | A6 | resnet50 | 10 | – | – | 0.596 [0.531, 0.658] | – |
| `models/final/N5_efficientnet_b0.pt` | N5 | efficientnet_b0 | 20 | 72.7 | 85.2 | 0.588 [0.531, 0.640] | 89.6 |
| `models/final/C2_efficientnet_b0.pt` | C2 | efficientnet_b0 | 56 | 71.6 | 84.7 | 0.588 [0.530, 0.641] | 92.1 |
| `models/final/N6_efficientnet_b0.pt` | N6 | efficientnet_b0 | 19 | 72.5 | 84.5 | 0.587 [0.530, 0.637] | 91.1 |
| `models/final/A1_resnet50.pt` | A1 | resnet50 | 8 | – | – | 0.587 [0.522, 0.648] | – |
| `models/final/X1_efficientnet_b0.pt` | X1 | efficientnet_b0 | 14 | 70.8 | 84.2 | 0.583 [0.518, 0.642] | 79.7 |
| `models/final/A3_efficientnet_b0.pt` | A3 | efficientnet_b0 | 12 | 71.5 | 85.1 | 0.581 [0.517, 0.640] | 78.4 |
| `models/coremil_final.pt` | A6 | efficientnet_b0 | 15 | 71.7 | 86.6 | 0.580 [0.518, 0.638] | 90.2 |
| `models/final/X4_efficientnet_b0.pt` | X4 | efficientnet_b0 | 18 | 71.3 | 85.6 | 0.579 [0.520, 0.633] | 92.6 |
| `models/final/A1_efficientnet_b0.pt` | A1 | efficientnet_b0 | 12 | 71.1 | 84.7 | 0.578 [0.516, 0.633] | 76.4 |
| `models/final/N3_efficientnet_b0.pt` | N3 | efficientnet_b0 | 11 | 70.9 | 85.1 | 0.576 [0.515, 0.634] | 78.9 |
| `models/final/A5_efficientnet_b0.pt` | A5 | efficientnet_b0 | 11 | 70.6 | 85.1 | 0.570 [0.507, 0.629] | 89.5 |
| `models/final/A6_convnext_tiny.pt` | A6 | convnext_tiny | 58 | – | – | 0.568 [0.502, 0.631] | – |
| `models/final/X2_efficientnet_b0.pt` | X2 | efficientnet_b0 | 21 | 70.8 | 86.7 | 0.568 [0.507, 0.625] | 87.2 |
| `models/final/A6_dinov2_small.pt` | A6 | dinov2_small | 58 | – | – | 0.564 [0.502, 0.620] | – |
| `models/final/B4_efficientnet_b0.pt` | B4 | efficientnet_b0 | 12 | 69.6 | 85.3 | 0.560 [0.498, 0.615] | 87.4 |
| `models/final/A2_efficientnet_b0.pt` | A2 | efficientnet_b0 | 10 | 69.4 | 84.1 | 0.559 [0.499, 0.618] | 88.5 |
| `models/final/A1_dinov2_small.pt` | A1 | dinov2_small | 21 | – | – | 0.559 [0.492, 0.617] | – |
| `models/final/A1_convnext_tiny.pt` | A1 | convnext_tiny | 14 | – | – | 0.537 [0.478, 0.590] | – |
