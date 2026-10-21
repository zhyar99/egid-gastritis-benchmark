# CoRe-MIL on EGID — technical report

Status: 2026-10-19. All numbers below come from `outputs/` (tables in `outputs/tables/{cv,test}`,
statistics in `outputs/metrics/{cv,test}`, figures in `outputs/figures`). Frozen-feature
experiments are complete; end-to-end fine-tuning is complete for 5 CV folds × 1 seed (§7.8).

## 1. Scientific motivation

EGID provides four patient-level gastritis labels for 229 patients but no per-image labels. The
reference models classify images independently and average their probabilities. We asked whether a
patient-level multiple-instance model that (i) pools images per task, (ii) respects the verified
label structure, (iii) is trained for incomplete image sets and (iv) is regularised by
confidence-gated cross-view consistency improves patient-level classification, robustness to
missing images and calibration. Background and building blocks: `literature_comparison.md`;
formal gap: `research_gap.md`.

## 2. Dataset (verified by audit)

- 229 patients, 5,883 unique images; the four task folders contain byte-identical copies
  (SHA-256) — one image set with four label views. Metadata agrees with folder membership for every
  patient and task; image counts equal `Number_of_pictures`; all images decode.
- 5,540 of the `.bmp` files are JPEG-encoded. 43 image sizes (431–1308 px wide). Bag size median
  24 (IQR 19–30, range 6–108).
- No cross-patient duplicates (pHash candidates confirmed by high-pass texture correlation;
  calibrated on 4 exact and 1 near-exact within-patient duplicates).
- **Presence of atrophy and gastritis type are exact functions of atrophy distribution** in all 229
  patients (normal→non-atrophic/normal; corpus-predominant→type A; antrum-predominant and
  antrum-and-corpus→type B). H. pylori is not determined by any label; all 31 corpus-predominant
  patients are H. pylori-negative (empirical association, not encoded).
- Class counts (patients): H. pylori +104/−125; distribution normal 64, antrum 74, corpus 31,
  antrum-and-corpus 60.

## 3. Model (summary; full formulation in `methodology.md`)

Frozen EfficientNet-B0 embeddings → MLP projection → task-specific gated attention (one branch per
independent output group, masked) → heads for H. pylori and distribution → dependent tasks via the
training-split mapping (structured head) → temperature scaling on validation patients.
Training: full-view and random-subset (20–80%) views, supervised loss on both, plus
confidence-gated (τ = 0.7), temperature-scaled (T = 2) KL from the detached full-view prediction to
the subset prediction, with a warm-up.

## 4. Validation design

- Development cohort 195 patients, 5-fold patient-level CV (stratified on H. pylori ×
  distribution), inner validation 15% of each training fold; results pooled over held-out folds
  (n = 195), 5 training seeds.
- Untouched test cohort of 34 patients (same size and class composition as the EGID paper's test
  cohort; different, because unpublished, patient IDs), evaluated once after the CV analysis.
- Pre-specified protocol (`experiment_protocol.md`), primary outcome: distribution macro-F1.
- Hyper-parameters chosen on inner-validation NLL only (one shared config for all MIL variants).
- Statistics: patient bootstrap (2,000), paired on identical resamples; McNemar on
  seed-ensembled predictions; seed SD and subset-draw SD reported separately.

## 5. Training procedure

AdamW (lr 3e-4, wd 1e-2), 16 patients per batch (256 images for image-level heads), dropout 0.5
(MIL) / 0.25 (image), input dropout 0.3, hidden 256, max 300 epochs, early stopping on validation
NLL (patience 25). Runtime per run on an M1 CPU: 9–50 s (Table 8). Feature extraction on MPS:
26–62 s per encoder for all 5,883 images.

## 6. Pre-specified results (CV, n = 195, 5 seeds)

Distribution macro-F1 with 95% patient-bootstrap CI (Table 3/5, Fig. 3/6):

| Model | Dist macro-F1 | Dist bal. acc | H. pylori macro-F1 | Mean macro-F1 (4 tasks) | Coherence |
|---|---|---|---|---|---|
| B1 image-level, ResNet50 | 0.672 [0.604, 0.734] | 0.642 | 0.876 | 0.713 | 0.839 |
| B2 image-level, EfficientNet-B0 | **0.722** [0.665, 0.773] | 0.706 | 0.875 | 0.727 | 0.790 |
| B3 image-level, DenseNet121 | **0.722** [0.663, 0.778] | 0.701 | 0.885 | **0.785** | 0.898 |
| A0 mean-pooling MIL | 0.677 [0.613, 0.731] | 0.674 | **0.904** | 0.760 | 0.889 |
| B4 gated-attention MIL | 0.560 [0.498, 0.615] | 0.567 | 0.853 | 0.680 | 0.874 |
| A1 task-aware MIL (B5) | 0.578 [0.516, 0.633] | 0.583 | 0.847 | 0.693 | 0.764 |
| A6 CoRe-MIL (B8) | 0.580 [0.518, 0.638] | 0.585 | 0.865 | 0.699 | 0.902 |

### RQ1 — task-aware MIL vs image-level aggregation: **not supported**
A1 − B2: −0.145 [−0.198, −0.089] (McNemar p = 0.001); A1 − A0: −0.099 [−0.145, −0.054]. All
attention-pooling variants were below both mean-pooling MIL and image-level aggregation. Diagnosis
(`outputs/history`, `outputs/attention`): attention models reach their best validation NLL at a
median epoch of 11–12 with training loss still 0.35 (mean pooling: epoch 51, loss 0.10) and then
overfit as attention concentrates (top weight ≈ 4.7× uniform for A1). With ~133 training patients
per fold and frozen features, learned instance weighting overfits; uniform weighting is a strong
inductive bias for bag labels that reflect *overall* findings. The task branches do learn different
weightings (A6, seed 0, held-out patients: median Spearman 0.44 between H. pylori and distribution
attention; distribution attention more peaked, mean normalised entropy 0.84 vs 0.94), but this did not translate into accuracy.

### RQ2 — structured head: **coherence improved, accuracy unchanged**
A2 − A1: −0.018 [−0.046, +0.009]. Coherence of the four per-task decisions rose from 0.764 (A1)
to 0.885 (A2) and 0.902 (A6); image-level models produced 10–21% incoherent label combinations
(e.g. B2 0.790). Probability-level consistency holds by construction; decision-level coherence does
not, because per-task argmax can disagree (methodology §2).

### RQ3 — view-dropout training and robustness: **small effect**
Dist macro-F1 at 25% of images (Table 6, Fig. 5): A1 0.537 → A3 0.543; A2 0.525 → A5 0.539
(A5 − A2 at 25%: +0.013 [+0.003, +0.024]). Degradation from 100% to 25% was similar with and
without view dropout (≈ 0.03–0.04). The most robust model was the image-level DenseNet121
(0.722 → 0.704).

### RQ4 — confidence-gated consistency beyond view dropout: **mixed**
Without structure: A4 − A3 = +0.024 [−0.003, +0.050] at 100% and +0.028 [+0.011, +0.045] at 25%
(A4 best epoch 35 vs 12: the term delays overfitting). With structure: A6 − A5 = +0.010
[−0.011, +0.030] (100%), +0.007 [−0.004, +0.018] (25%). The effect is small and not consistent.

### RQ5 — calibration: **temperature scaling did not help MIL models**
Dist ECE before → after temperature scaling (pooled CV, Table 7a, Fig. 7): B2 0.159 → 0.057,
B1 0.097 → 0.051, B3 0.078 → 0.076, A0 0.071 → 0.062, A1 0.094 → 0.091, A6 0.071 → 0.091.
For MIL models fitted temperatures on ~23 validation patients did not transfer reliably; early
stopping on validation NLL had already tuned confidence. NLL ordering mirrors accuracy (B2 0.781,
A0 0.847, A6 1.040). The rare corpus-predominant class was not the hard one (recall 0.78–0.88 for
all models); errors concentrate between antrum-predominant and antrum-and-corpus (Table 4, Fig. 4).

### Loss and sampling variants (Table 5)
Focal loss improved CoRe-MIL (X3 0.641 [0.582, 0.694]; X3 − A6 = +0.061 [+0.033, +0.090]),
consistent with the overfitting diagnosis (focal loss down-weights confidently fitted patients).
Class-weighted CE (X2 0.568), balanced patient sampling (X4 0.579) and the soft clinical loss
(X1 0.583) did not help.

## 7. Further analyses

### 7.1 Post-hoc exploratory: CoRe-MIL components on mean pooling
Defined after the CV results (protocol §9); reported separately. Mean pooling + structure + view
dropout + consistency (M6): 0.686 [0.621, 0.741]; M6 − A0: +0.009 [−0.015, +0.034] at 100%,
+0.022 [+0.004, +0.041] at 50%, +0.026 [+0.010, +0.043] at 25%; coherence 0.961. M6 − B2:
−0.036 [−0.082, +0.009] at 100%. Interpretation: the robustness and coherence components give
small, consistent gains when pooling does not overfit, but do not close the gap to image-level
aggregation in CV.

### 7.2 Test cohort (n = 34, evaluated once)
Dist macro-F1: B3 0.669 [0.455, 0.816], B1 0.634, B2 0.567, A0 0.606, A1 0.525, A6 0.575,
A4 0.612, X3 0.615, M6 0.739 [0.601, 0.848]. H. pylori macro-F1 0.90–1.00 (B3 1.000).
Pre-specified A6 − B3: −0.094 [−0.242, +0.075]. CIs span ~0.3–0.4; the test ranking differs from
CV (e.g. B2 drops from 0.722 to 0.567; post-hoc M6 is highest on test but below B2/B3 in CV;
M6 − B3 on test +0.070 [−0.080, +0.272]). The test cohort is too small to rank models; CV is the
primary evidence. The EGID paper's DenseNet121 test macro-F1 for distribution (0.761) is from a
different 34-patient cohort and a fine-tuned model; it is not comparable to any number here.

### 7.3 Encoder sweep (CV, 3–5 seeds; Table S1)
Distribution macro-F1, IMG / A0 / A1 / A6: EfficientNet-B0 0.722 / 0.677 / 0.578 / 0.580;
DenseNet121 0.722 / 0.692 / 0.625 / 0.632; ResNet50 0.672 / 0.643 / 0.587 / 0.596; ConvNeXt-T
0.697 / 0.628 / 0.537 / 0.568; Swin-T 0.727 / 0.705 / 0.627 / 0.668; DINOv2-S 0.693 / 0.667 /
0.559 / 0.564. The ordering IMG ≥ A0 > A6 ≥ A1 held for every encoder; the attention gap is
smallest with Swin-T. The conclusion is not an artefact of the pre-specified encoder.

### 7.4 Missing-view robustness (Table 6, Fig. 5)
All models degrade monotonically. 100% → 25%: B3 −0.018, B2 −0.067, A0 −0.075, M6 −0.058,
A1 −0.041, A6 −0.035. Attention models lose less in absolute terms but start much lower.
Between-draw SDs (≤ 0.01) are small relative to between-seed SDs and patient-bootstrap CIs.

### 7.5 Interpretability (Fig. 8)
Highest-weighted images per branch for held-out patients are shown for one correct and two
incorrect cases. Attention indicates which images the pooling relied on; it is not lesion
localisation and was not validated against any ground truth. Grad-CAM is not meaningful for the
frozen workflow (the encoder is not trained, so its saliency would not explain the MIL decision); it
is shown for the end-to-end fine-tuned CoRe-MIL instead (supplementary
`outputs/figures/cv/figS_gradcam_ft_a6.*`, `scripts/gradcam.py`): gradients of the patient-level
distribution logit w.r.t. each image's last 7×7 feature map, for the four most-attended images of
one correct and one misclassified cv0 patient. The maps fall on mucosa rather than on the black
mask, but they are coarse and unvalidated and do not establish lesion localisation.

### 7.6 Statistical summary
Patient-sampling uncertainty dominates: CV CIs ≈ ±0.06, test CIs ≈ ±0.15–0.20, seed SDs
0.006–0.028, subset-draw SDs ≤ 0.01. All p-values are unadjusted and descriptive
(`outputs/metrics/*/paired_comparisons.csv`).

### 7.7 Compute (Table 8)
Cached-feature runs: 9–50 s each on CPU; the frozen-feature programme (≈ 840 runs) took roughly
2 h wall-clock with 5–10 parallel processes on the M1, plus ≈ 1.5 h for the tuning grid (480 fits). Fine-tuning DenseNet121 image-level:
≈ 75 s per epoch on MPS without contention, 30–75 min per fold.

### 7.8 End-to-end fine-tuning
**FT_IMG** (DenseNet121 fine-tuned image-level multitask CNN following the EGID paper's described
protocol: ≤ 25 epochs, early stopping with patience 5 on validation mean accuracy, augmentation,
mean-probability aggregation; unreported settings chosen by us), 5 CV folds, **1 seed**:
distribution macro-F1 0.614 [0.536, 0.685]; H. pylori 0.859 [0.808, 0.906]; presence 0.753;
type 0.661. Per-fold values ranged 0.46–0.76, and the accuracy-based early stopping on ~23
validation patients selected epoch 1 in one fold. Paired with A6: A6 − FT_IMG = −0.034
[−0.109, +0.044]. Fine-tuning under this protocol did **not** beat frozen features with a tuned
head (0.722); with one seed this is indicative only.
**FT_A6** (end-to-end CoRe-MIL, EfficientNet-B0; 12 images per patient per step, 2 patients × 4
accumulation steps, backbone lr 5e-5, early stopping on validation NLL), 5 CV folds, **1 seed**:
distribution macro-F1 0.609 [0.533, 0.678] (per fold 0.38–0.70); at 25% of images 0.546;
H. pylori 0.812; mean of four tasks 0.690; coherence 0.933. Versus frozen A6: A6 − FT_A6 = −0.029
[−0.108, +0.052] at 100% and −0.000 [−0.046, +0.047] at 25%. ≈ 37 s/epoch on MPS, 11–19 min per
fold. End-to-end training therefore did not rescue the attention-based model: it remains below the
frozen-feature image-level baselines (0.722), and neither fine-tuned model separates from the other
with one seed (FT_A6 0.609 vs FT_IMG 0.614).

**FT_A6, 3 seeds** (seeds 1–2 added; 15 CV runs): pooled distribution macro-F1 0.617 [0.558, 0.671]
(per-seed mean 0.596 / 0.613 / 0.615; per run 0.38–0.73, fold cv2 worst, cv3 best); mean accuracy
71.8%; coherence 92.8%. FT_A6 − A6 = +0.037 [−0.030, +0.101]; FT_A6 − B3 = −0.105 [−0.166, −0.045].
Test cohort (34 patients, 3 seeds): 0.563 [0.422, 0.675]. Exported on all 229 patients as
`models/coremil_ft_final.pt` (20 epochs).

**Addendum A — FT2_IMG** (improved fine-tuning recipe, pre-registered, protocol §10; 2 seeds).
Pooled CV distribution macro-F1, fine-tuned − frozen with the same encoder (primary comparison):

| Encoder | FT2_IMG | Frozen IMG | Difference [95% CI] | Test cohort FT2_IMG |
|---|---|---|---|---|
| DenseNet121 | 0.703 [0.646, 0.756] | 0.722 | −0.019 [−0.073, +0.032] | 0.593 [0.389, 0.734] |
| EfficientNet-B0 | 0.730 [0.671, 0.782] | 0.722 | +0.008 [−0.041, +0.056] | 0.643 [0.448, 0.777] |
| ResNet50 | 0.648 [0.581, 0.706] | 0.672 | −0.024 [−0.083, +0.033] | 0.589 [0.396, 0.724] |

Validation-NLL early stopping removed the epoch-1 failures of FT_IMG, and all three fine-tuned CNNs beat
CoRe-MIL, but none differs from its frozen counterpart. FT2 EfficientNet-B0 has the highest CV point
estimate of the study (0.730), not separable from the frozen image-level models. Exploratory (folds reused).


### 7.9 Addendum B — stronger MIL aggregators (pre-registered, protocol §11)
Hyper-parameters per family selected on inner validation (N1: hidden 128, dropout 0.5, input dropout 0;
N3: dropout 0.25, weight decay 0.1; N5: hidden 128, dropout 0.5). Pooled CV distribution macro-F1:
N1 top-k 0.616 [0.554, 0.672]; N2 top-k + structure 0.611 [0.551, 0.663]; N3 task-aware attention + ACMIL masking 0.576 [0.515, 0.634];
N4 CoRe-MIL + masking 0.626 [0.568, 0.679]; N5 transformer 0.588 [0.531, 0.640]; N6 transformer + structure 0.587 [0.530, 0.637].
Paired: N1 − B3 -0.106 [-0.167, -0.047]; N3 − B3 -0.145 [-0.210, -0.077]; N5 − B3 -0.134 [-0.195, -0.073]; N1 − A0 -0.061 [-0.102, -0.021];
N3 − A1 -0.001 [-0.024, +0.019]; N4 − A6 +0.046 [+0.019, +0.074]. By the pre-registered decision rule the conclusion is unchanged:
no MIL aggregator outperforms image-level averaging, and all are below mean pooling. Masking helps only in
combination with the other components (N4). Test cohort: N1 0.548 [0.377, 0.683], N4 0.554 [0.414, 0.663], N5 0.535 [0.396, 0.634].

### 7.10 Addendum C — image-level auxiliary loss and augmented features (pre-registered, protocol §12)
CV only, 5 folds × 5 seeds, A6 hyper-parameters unchanged. Pooled CV (195 patients):

| Model | Distribution macro-F1 | Mean accuracy | Coherence | Dist. macro-F1, 25% of images |
|---|---|---|---|---|
| A6 CoRe-MIL | 0.580 [0.518, 0.638] | 71.7% | 90.2% | 0.545 |
| C1 + image-level loss (λ = 0.5) | 0.637 [0.579, 0.689] | 74.0% | 93.6% | 0.590 |
| C2 + augmented features (10 views) | 0.588 [0.530, 0.641] | 71.6% | 92.1% | 0.546 |
| C3 + both | 0.607 [0.550, 0.658] | 72.0% | 92.8% | 0.563 |

Paired differences (distribution macro-F1): **primary C3 − A6 = +0.027 [−0.009, +0.063]**; C1 − A6 = +0.057
[+0.025, +0.086]; C2 − A6 = +0.008 [−0.033, +0.049]; C3 − B2 = −0.115 [−0.168, −0.065]; C3 − B3 = −0.115
[−0.176, −0.056]. Under the pre-registered rule the primary comparison does **not** establish an improved
CoRe-MIL (interval contains zero), and C3 remains below image-level averaging, so the manuscript conclusion
is unchanged. As a secondary, exploratory finding, the image-level loss alone (C1) improved A6 with an interval
above zero and moved the best epoch from a median of 14 to 44, consistent with less overfitting; augmenting
frozen features had no measurable effect and diluted the gain when combined. C1 remains below mean-pooling
CoRe-MIL (M6, 0.686) and image-level averaging (0.722); C1 is not the pre-registered primary model and is not
promoted post hoc.

## 8. Conclusions

1. On EGID with frozen ImageNet features, **learned attention pooling (and therefore the
   pre-specified CoRe-MIL, A6) did not improve on mean-probability image-level aggregation or
   mean-pooling MIL**; it was clearly worse in CV (≈ −0.10 to −0.14 macro-F1). The hypothesis of
   RQ1 is rejected in this setting.
2. The **structured head** is a cheap way to obtain more coherent multi-task outputs without
   loss of accuracy, and it removes the redundancy of modelling four labels when two determine all.
3. **View dropout and confidence-gated consistency** produce small robustness gains, most visible at
   25–50% of images and when combined with mean pooling (post hoc); effects are within ~0.03.
4. Temperature scaling on tiny validation sets is unreliable for MIL models here.
5. Strongest frozen-feature baselines: image-level aggregation with DenseNet121/EfficientNet-B0/
   Swin-T (0.72–0.73 CV distribution macro-F1); mean-pooling MIL is best for H. pylori (0.904).

6. End-to-end fine-tuning did not change these conclusions: FT_A6 (3 seeds) 0.617 and the
   EGID-protocol FT_IMG 0.614 were below frozen image-level aggregation, and the improved recipe
   (Addendum A, 2 seeds) matched but did not exceed frozen features for any CNN (0.648–0.730).

What would be needed to test CoRe-MIL fairly as a deployable model: multi-seed fine-tuning with
tuned schedules, attention regularisation (e.g. entropy penalties, attention dropout) or pre-training,
a larger multi-centre cohort, and per-image anatomical labels for true missing-view experiments.

## 9. Limitations and reproducibility

See `limitations.md` and the README. Every run has a JSON record with seed, split, configuration
(and its hash), metrics, temperatures, mapping, checkpoint, runtime, library versions and code hash.
