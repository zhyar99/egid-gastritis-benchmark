# Experiment protocol (pre-specified)

Frozen on 2026-10-19, **before** any main-grid run and before any evaluation of the primary
test cohort. Hyper-parameter selection (inner-validation NLL only, `scripts/tune.py`) and the
smoke/sanity runs preceded this freeze; no CV held-out fold or test patient informed it.
Any later deviation is listed in §9 with its reason.

## 1. Data and partitions

| Partition | Patients | Use |
|---|---|---|
| Primary train / val / test | 161 / 34 / 34 | Same cohort sizes and per-cohort class counts as EGID Table 3 (Li et al. 2026). Patient IDs of the original split were not published, so ours differ: **published metrics are not directly comparable.** |
| Development cohort | 195 (primary train + val) | 5-fold patient-level CV (`cv0..cv4`), stratified on joint H. pylori × distribution label; inner validation = 15% of each training fold (stratified). |
| Primary test | 34 | Evaluated **once**, after all CV analyses, with `--allow-test`. Never used for selection. |

Units of splitting are audit split-groups (patients linked by cross-patient duplicate images); with
the current data every group is one patient (no cross-patient duplicates found). All variants use
identical split files (`outputs/splits/*.json`).

## 2. Label structure (verified, not assumed)

The audit (`outputs/data_audit/dataset_audit.md`) shows that, in all 229 patients, *presence of
atrophy* and *gastritis type* are exact functions of *atrophy distribution*. The structured head
therefore predicts {H. pylori, distribution} and derives the other two tasks; the mapping is
re-derived from the **training patients of each split** and the structured head is used only if it
is exactly deterministic there. Results for the two derived tasks are not reported as independent
clinical findings. H. pylori is not determined by any label and is predicted independently; the
empirical `corpus_predominant ⇒ H. pylori negative` association (31/31) is **not** encoded.

## 3. Outcomes

- **Primary outcome:** patient-level macro-F1 for atrophy distribution (4 classes), on calibrated
  probabilities (argmax), pooled over the 5 CV held-out folds (n = 195) and averaged over seeds.
  The same metric on the primary test cohort (n = 34) is a confirmatory secondary analysis.
- **Secondary:** per-task macro-F1, balanced accuracy, accuracy, AUROC/AUPRC (OvR macro; undefined
  when a class is absent), NLL, Brier, ECE (10 bins, top-label); mean over the four tasks;
  prediction coherence rate; per-class sensitivity for distribution; missing-view degradation;
  compute cost.

## 4. Model selection rules

- Early stopping and checkpointing: lowest mean patient-level NLL (4 tasks) on inner-validation
  patients; patience 25 epochs; never stop while the consistency term is ramping in.
- Hyper-parameters: one config for image-level heads (selected on IMG) and **one shared config for
  all MIL variants** (selected on the mean of A0 and A1), from a 32-point grid, by mean
  inner-validation NLL over the 5 folds (seed 0). The proposed components (A2–A6) are not tuned,
  which is conservative with respect to the proposed model.
- Consistency hyper-parameters fixed a priori: λ2 = 1, T = 2, τ = 0.7, warm-up start 5, ramp 15
  epochs; student ratio U[0.2, 0.8]; teacher = full bag.
- Temperature scaling: one scalar per independent output group, fitted on inner-validation patients.
- Primary encoder: EfficientNet-B0 (pre-specified). The encoder sweep is a secondary analysis.

## 5. Hypotheses

| RQ | Hypothesis (H1 = alternative) | Test |
|---|---|---|
| RQ1 | H1a: task-aware MIL (A1) has higher distribution macro-F1 than image-level mean-probability aggregation with the same frozen encoder (B2 IMG). | Paired patient bootstrap Δ (A1 − B2), 95% CI; McNemar on seed-ensembled predictions. |
| RQ2 | H2a: the structured head (A2) yields a higher coherence rate of per-task decisions than independent heads (A1) (probability-level consistency holds by construction; decision-level coherence does not). H2b: A2 ≥ A1 in distribution macro-F1. | Coherence rate; paired bootstrap Δ (A2 − A1). |
| RQ3 | H3: view-dropout training (A3 vs A1; A5 vs A2) reduces the drop in distribution macro-F1 from 100% to 25% images. | Degradation curves; paired Δ at 25% and 50%. |
| RQ4 | H4: confidence-gated consistency adds benefit beyond view dropout (A6 vs A5; A4 vs A3), at 100% and at 25%. | Paired bootstrap Δ. |
| RQ5 | H5: temperature scaling lowers distribution ECE/NLL for all models; CoRe-MIL is not worse calibrated than baselines. | ECE/NLL before vs after; reliability diagrams; per-class recall for the rare corpus-predominant class. |

Null results are reported as such. With ~195 CV patients (31 corpus-predominant) and 34 test
patients, small differences (|Δ macro-F1| < ~0.05) are not expected to be resolvable; CIs are
reported instead of significance claims from single runs. p-values are unadjusted and
interpreted descriptively; the family of ablation comparisons is exploratory.

## 6. Experimental grid

- `full` profile, CV splits: all 13 frozen-feature variants × EfficientNet-B0 × 5 folds × 5 seeds.
- Baselines B1/B3: IMG with ResNet50 / DenseNet121 features, 5 folds × 5 seeds.
- Encoder sweep (secondary): IMG, A0, A1, A6 × 6 encoders × 5 folds × 3 seeds.
- Missing-view evaluation: fractions {1.0, 0.75, 0.5, 0.25}, 10 subset draws per fraction (5 for the
  encoder sweep); subsets depend only on (patient, fraction, draw), hence identical across models.
- Primary test: the same variants, 5 seeds, trained on the 161 training patients (34 validation).
- Optional fine-tuning (FT_IMG with DenseNet121 following the EGID paper protocol; FT_A6 with
  EfficientNet-B0): as many CV folds as runtime allows; reported as exploratory.

## 7. Statistical analysis

- Patients are the resampling unit (never images). 2,000 bootstrap resamples.
- Uncertainty sources reported separately: patient sampling (bootstrap CI), training seed
  (between-seed SD), image-subset sampling (between-draw SD).
- Paired comparisons use identical patient resamples for both models (seed-averaged metric).

## 8. What would count as support

CoRe-MIL would be supported if (i) A6 − B2 and A6 − A0 distribution macro-F1 CIs exclude 0 in CV,
(ii) the advantage persists or grows at 25–50% images, and (iii) the direction replicates on the
test cohort. Anything weaker is reported as inconclusive.

## 9. Deviations log

- 2026-10-19: before the freeze, sanity runs showed rapid overfitting of attention MIL with the
  initial defaults (best epoch ≈ 5). Added an input-dropout option and the inner-validation grid
  search (§4). These runs were deleted and are not part of any reported result.
- 2026-10-19 (after the pre-specified CV grid, before any test-set run): all attention-based variants
  (B4, A1–A6) were clearly below image-level aggregation and mean-pooling MIL in CV. **Post-hoc,
  exploratory** variants M2/M3/M5/M6 apply the same CoRe-MIL components (structured head, view
  dropout, cross-view consistency) to mean pooling, with the unchanged shared MIL hyper-parameters.
  They were defined from CV results only; they are reported separately and do not replace the
  pre-specified comparisons.

## 10. Addendum A — improved CNN fine-tuning (pre-registered 2026-10-19, before any FT2 run)

Motivation: the EGID-protocol fine-tuning (FT_IMG, accuracy-based early stopping) was erratic
(best epoch 1–12; selected epoch 1 in one fold). This addendum tests whether a better, fixed recipe
lets end-to-end CNNs reach or exceed the frozen-feature image-level baseline.

- Variant **FT2_IMG**: image-level multitask CNN, all layers trainable, linear heads with dropout 0.2,
  mean-probability patient aggregation (EGID protocol). Recipe fixed a priori, not tuned:
  AdamW, backbone lr 1e-4, head lr 1e-3, weight decay 1e-4, batch 16 images, augmentation as FT_IMG,
  max 30 epochs, early stopping on **validation patient-level NLL** (patience 6), temperature scaling
  on validation patients. The recipe was chosen from FT_IMG inner-validation curves only.
- Encoders: DenseNet121, EfficientNet-B0, ResNet50 (all CNNs). Seeds 0 and 1.
- Splits: cv0–cv4 (primary analysis, pooled n = 195) and the primary split (34 test patients).
- Primary comparison: FT2_IMG vs frozen-feature IMG with the same encoder, distribution macro-F1,
  pooled CV, paired patient bootstrap. Secondary: vs CoRe-MIL (A6), test cohort, all other metrics.
- Status: exploratory relative to the original protocol (the CV folds and test cohort were already
  used for other models); reported as such.

## 11. Addendum B — stronger MIL baselines (pre-registered 2026-10-19, before any N-variant run)

Motivation: reviewers may regard the attention models as under-specified strawmen. This addendum adds
three established aggregation families on the same cached EfficientNet-B0 features and partitions.

| ID | Aggregation | Output |
|---|---|---|
| N1 | Top-k MIL: per-image class logits, patient logit per class = mean of the k = 7 highest image logits (k = min(7, N); HOPE AI, Hu et al. 2025) | Independent |
| N2 | As N1 | Structured |
| N3 | Task-aware gated attention with ACMIL-style stochastic top-instance masking during training: in each branch the 5 highest-scoring images are masked with p = 0.5 each, keeping ≥ 1 image (Zhang et al. 2025) | Independent |
| N4 | As N3 + structured head + image-subset training + cross-view consistency (CoRe-MIL with masking) | Structured |
| N5 | Transformer MIL: 2 pre-norm self-attention layers (4 heads, FFN 512) over the projected image features with a learnable class token, following TransMIL (Shao et al. 2021) without its positional module because a patient's images have no spatial order | Independent |
| N6 | As N5 | Structured |

- Hyper-parameters: for each family the 32-point grid of §4 (learning rate, dropout, weight decay, hidden size,
  input dropout) is searched on N1, N3 and N5 by mean inner-validation NLL over the five folds (seed 0);
  the structured variants reuse their family's selection. k, the masking settings and the transformer
  depth/width are fixed a priori as above. No held-out fold or test patient is used for selection.
- Runs: 5 CV folds × 5 seeds and the primary split × 5 seeds, identical to the main grid.
- Primary comparison: distribution macro-F1 (pooled CV) of N1, N3 and N5 versus image-level DenseNet121
  (B3) and versus mean-pooling MIL (A0), paired patient bootstrap. Secondary: N3 vs A1 (effect of masking),
  N4 vs A6, N2/N6 vs N1/N5 (structured head), coherence, 25%-image robustness, calibration, test cohort.
- Decision rule stated in advance: the manuscript conclusion "no evaluated MIL aggregator outperforms
  image-level averaging" is revised if any N-variant has a paired CV difference to B3 whose 95% interval
  lies entirely above zero; it is weakened to "comparable" if any interval contains zero with a positive
  point estimate.

## 12. Addendum C — image-level supervision and feature augmentation for CoRe-MIL (pre-registered 2026-10-20, before any C-variant run)

Motivation: in the main grid the attention-based CoRe-MIL (A6) overfitted its training patients
(training loss ≈ 0.15 versus validation NLL ≈ 0.55; `outputs/figures/cv/figS_training_curves.png`),
whereas image-level training, where every image carries its patient's labels, gave the best results.
Frozen-feature models also saw no image augmentation. This addendum tests two remedies on top of A6,
with every other setting unchanged.

| ID | Change relative to A6 |
|---|---|
| C1 | + image-level auxiliary loss: every image of the full training view is scored by the same task heads as the patient (shared linear heads on the image embedding, structured mapping applied), and the supervised loss of the image predictions against the patient's labels is added with weight λ_img = 0.5 |
| C2 | + augmented feature views: 10 augmented copies of every image are encoded once with the frozen EfficientNet-B0 (augmentation as in fine-tuning: random resized crop 224, scale 0.75–1, ratio 0.9–1.11; horizontal flip; rotation ±15°; brightness/contrast 0.15, saturation 0.1; no hue change; fixed seeds). In each training epoch every image is represented by one of its 11 versions (clean + 10 augmented), drawn uniformly; validation, temperature fitting and evaluation use the clean features only |
| C3 | C1 + C2 |

- Hyper-parameters: those of A6 (shared tuned MIL configuration of §4); λ_img = 0.5 and the 10 augmented copies are
  fixed a priori and not tuned. Feature standardisation uses the clean training-image statistics for all versions.
- Runs: 5 CV folds × 5 seeds (seeds 0–4, profile `full`, same partitions as A6). The primary split / test cohort is
  not part of this addendum; if it is run later, it is run once for C3 only and reported separately.
- Primary comparison: C3 vs A6, distribution macro-F1, pooled CV, paired patient bootstrap (2,000 resamples).
- Secondary: C1 vs A6 and C2 vs A6 (each remedy alone); C3 vs image-level EfficientNet-B0 (B2, same encoder) and
  vs image-level DenseNet121 (B3); coherence, 25%-image robustness, calibration.
- Decision rule stated in advance: an "improved CoRe-MIL" is claimed only if the 95% interval of C3 − A6 lies
  entirely above zero. C3 is said to outperform image-level averaging only if the interval of C3 − B3 lies entirely
  above zero, and to be comparable if that interval contains zero; otherwise the manuscript conclusion is unchanged.
- Status: exploratory (the CV folds were already used for other models); reported in full whatever the outcome.
