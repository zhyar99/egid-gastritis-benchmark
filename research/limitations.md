# Limitations

## Data

1. **Single centre, single device, retrospective, curated.** Renji Hospital, Olympus CV-290,
   white light, images pre-selected for quality; cardia and pylorus excluded. No external or
   prospective validation exists, so no result here supports clinical deployment.
2. **Small cohort.** 229 patients; 31 corpus-predominant / type A patients. In 5-fold CV each
   held-out fold has 5–6 corpus-predominant patients; the 34-patient test cohort has 5. Per-class
   estimates for this class are imprecise regardless of model.
3. **Patient-level labels only.** No per-image labels or anatomical locations; attention cannot be
   validated against ground-truth regions, and missingness experiments are random image subsets,
   not anatomically defined missing views.
4. **Label redundancy.** Presence of atrophy and gastritis type are deterministic functions of
   atrophy distribution in these annotations. Performance on those two tasks is not independent
   evidence; "four-task" means are dominated by two independent dimensions (H. pylori,
   distribution). The structured head exploits a property of EGID's annotation scheme that may not
   transfer to cohorts labelled differently.
5. **Label source differs by task.** H. pylori status comes from rapid urease testing (a test with
   its own error rate) and is not image-adjudicated; the other three come from endoscopist consensus
   on the same images the model sees (κ 0.947–0.970), which may favour image-based models.
6. **Container format.** 5,540 of the 5,883 `.bmp` files are JPEG-encoded; images have therefore
   been lossy-compressed at some point despite the "BMP to preserve quality" description.
7. **Within-patient exact duplicates.** 4 byte-identical pairs and 1 near-identical pair exist
   within patients (no cross-patient duplicates). They slightly up-weight those views and were kept
   as released.

## Comparability with the EGID paper

8. The published split's patient IDs, seed, image size, optimiser and learning rate are not
   reported. Our primary split matches cohort sizes and per-cohort class counts exactly but
   contains different patients. **Our numbers must not be placed in the same table as the published
   ones without this caveat.** Our B1–B3 baselines on frozen features are *frozen-feature
   variants of* the published fine-tuned models, not reproductions; FT_IMG (DenseNet121) follows
   the described protocol (≤25 epochs, patience 5 on validation mean accuracy, standard
   augmentation, mean-probability aggregation) with our own choices for unreported settings
   (AdamW, lr 1e-4, batch 16, 224 px).

## Method and evaluation

9. **Frozen ImageNet features** are the main workflow for runtime reasons on an M1. Domain shift
   to endoscopy is large. End-to-end fine-tuning was run for FT_IMG (1 seed), FT_A6 (3 seeds)
   and the pre-registered FT2_IMG recipe (3 CNNs, 2 seeds) with untuned, a-priori schedules;
   with so few seeds, fine-tuning conclusions are indicative only.
10. **Hyper-parameter selection** used inner-validation NLL with ~23 patients per fold, which is
    noisy; one shared MIL config was selected on A0/A1 and applied unchanged to A2–A6 (conservative
    for the proposed components but possibly sub-optimal for them). Consistency hyper-parameters
    (λ2, T, τ, warm-up, student ratio) were not tuned.
11. **Early stopping on ~23 validation patients** (CV) or 34 (primary) introduces selection noise;
    multiple seeds partly average it out.
12. **Statistical power.** With 195 pooled CV patients, macro-F1 CIs are roughly ±0.06–0.08; with 34
    test patients roughly ±0.15. Differences between close variants are generally not resolvable.
    p-values are unadjusted for the number of comparisons.
13. **CV pooling.** Out-of-fold predictions from five differently trained models are pooled
    before computing metrics; calibration of pooled predictions mixes five temperature fits.
14. **Attention ≠ localisation.** Attention weights show what the pooling relied on, not where
    pathology is; no pixel-level explanation is validated. Grad-CAM is shown only for the
    fine-tuned FT_A6 (frozen encoders are not trained, so their saliency does not explain the MIL
    head); its 7×7 maps are coarse and were not compared with any lesion annotation.
15. **Calibration** is assessed with ECE on ≤195 patients and 10 bins; ECE is biased and high-variance
    at this sample size; NLL and Brier are reported alongside.
16. **Literature search is single-reviewer** (`literature_search.md`, run 2026-10-19). Scopus,
    Embase and Web of Science were not queried (OpenAlex used instead); Google Scholar and CNKI
    coverage is partial. Full texts were read for 4 of the 10 closest papers; 6 are paywalled and
    rest on abstracts. Novelty boundaries: `research_gap.md` §4.1; open items:
    `literature_search.md` §8.
