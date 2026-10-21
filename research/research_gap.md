# Formal research gap

## Status of this document

This gap statement is derived from (i) the EGID data descriptor (Li et al., *Sci Data* 13:1396,
2026), (ii) the verified properties of the released data (`outputs/data_audit/dataset_audit.md`)
and (iii) a systematic search run on 2026-10-19 (`literature_search.md`: PubMed, arXiv, IEEE
Xplore, OpenAlex in place of Scopus, plus Google Scholar (partial), medRxiv and CNKI; 50 studies
included), summarised in `literature_comparison.md`. The search shows that several things this
document originally treated as open are **not** open in the wider literature (§4). Statements of
the form "X has not been studied" remain restricted to *EGID* unless §4 says otherwise. Claims
about what other studies did not do are tagged **[FT]** (checked in the full text: #2, #5, #8,
#9) or **[abs]** (abstract only: #1, #3, #4, #6, #7, #10, which are paywalled). The [abs] claims
must be confirmed before submission (`literature_search.md` §7.7, §8).

## 1. What the EGID benchmark establishes

- 229 patients, 5,883 images, four patient-level label dimensions; labels apply to the patient, not
  to individual images.
- Reference technical validation: image-level multitask CNNs (ResNet18/50, EfficientNet-B0,
  DenseNet121), fine-tuned, with patient prediction = mean of image softmax probabilities; best
  DenseNet121 with mean patient-level accuracy 83.82% and distribution accuracy 73.53% on a
  34-patient test cohort (95% CI for distribution accuracy 58.75–88.24%).
- Patient IDs of the reference split are not published.

## 2. Gaps (specific to EGID, verifiable)

**G1 — Weak supervision is handled by label copying.** Every image inherits its patient's labels,
although the descriptor states that labels reflect overall findings across gastric regions, and
images of the corpus and of the antrum carry different evidence for *distribution* of atrophy.
Image-level training with copied labels treats uninformative views as positive evidence. A
patient-level multiple-instance formulation, with pooling learned per task, has not been evaluated
on EGID. (Patient-level MIL on gastroscopy itself is established: for H. pylori by PhenoMIL
[abs], HOPE AI [FT] and Ting et al. 2025 [abs]; for precancerous triage by EndoStrat [FT], which
uses the same gated-attention pooling as our A1. The gap is EGID- and task-specific, not
methodological.)

**G2 — The label dimensions are not independent, but were modelled as if they were.** The
descriptor calls the four dimensions "independent", yet the released annotations make *presence of
atrophy* and *gastritis type* exact functions of *distribution of atrophy* in all 229 patients.
Four independent heads can therefore output clinically impossible combinations (e.g. "non-atrophic"
with "type B"), and the reported per-task accuracies (e.g. 82.35% presence vs 85.29% type vs
73.53% distribution for DenseNet121) are mutually inconsistent in a way a coherent predictor
could not produce. The effective learning problem is two-dimensional (H. pylori × distribution);
whether exploiting this structure helps, and how often unstructured models are incoherent, is
unmeasured. (Label-consistency methods are established in medical imaging, e.g. C²Net for
fundus and hierarchical CXR classification. GSCNet [FT] encodes the antrum-vs-corpus definition
of corpus-predominant gastritis in losses, but takes a fixed triplet of section-specific images.
Cao et al. [FT] classify region and atrophy per image from image-level labels and report no
patient-level diagnosis. No included gastritis study reports a coherence rate: [FT] for #2, #5,
#8, #9; [abs] for the rest.)

**G3 — Robustness to the number of available images is unknown.** Bag sizes vary 18-fold (6–108
images). In practice an examination may contain fewer usable frames than in the curated dataset.
No EGID result reports how patient-level performance degrades when only a fraction of the images is
available, and no model is trained for it. (Training on random instance subsets and
full-bag→masked-bag consistency are established in WSI MIL — DTFD-MIL pseudo-bags, MHIM-MIL,
ACMIL, PseMix. On gastroscopy, HOPE AI and EndoStrat [FT] require a minimum number of images
(≥ 8 and ≥ 20) and never test fewer; GSCNet [FT] cannot run without all three sections; Cao et
al. [FT] is image-level. Dai J/Dai D [abs] compare one image or site against all, which is not a
degradation curve. The remaining papers are [abs].)

**G4 — Calibration and uncertainty are not reported.** Only discrimination metrics with bootstrap
CIs are reported. For a 34-patient test cohort with 5 corpus-predominant patients, calibration
(ECE, NLL, Brier) and per-class sensitivity of the rare class are needed to judge clinical
usefulness; post-hoc calibration has not been assessed.

**G5 — Evaluation variance is dominated by patient sampling.** A single 34-patient test split gives
CIs ~±15 accuracy points. Patient-level cross-validation with repeated seeds and separate
accounting of seed, patient and image-subset variability is absent.

## 3. Research questions and the proposed answer

CoRe-MIL addresses G1–G5 with an integration of known building blocks; the contribution claimed is
the **formulation for this clinical structure and its controlled evaluation**, not any single
component:

| Gap | Component | Evaluated by |
|---|---|---|
| G1 | Patient bags; task-specific gated attention (one branch per independent output group) | RQ1: A1 vs image-level (B1–B3) and mean pooling (A0) |
| G2 | Structured head predicting {H. pylori, distribution}; dependent tasks via a mapping verified on training patients | RQ2: coherence rate, A2 vs A1 |
| G3 | View-subset training; confidence-gated cross-view consistency with a detached full-view teacher | RQ3/RQ4: A3/A5 vs A1/A2; A6 vs A5; degradation at 75/50/25% images |
| G4 | Validation-fitted temperature scaling; confidence/entropy outputs | RQ5: ECE/NLL/Brier, reliability diagrams, rare-class recall |
| G5 | 5-fold patient-level CV × multiple seeds + untouched test cohort; patient bootstrap | All RQs |

## 4. Boundaries of the claim

### 4.1 Novelty boundaries after the systematic search (2026-10-19)

Not claimable (prior art found; numbers refer to `literature_search.md` §7):

- "First MIL / patient-level model for gastroscopy or H. pylori": #1 [abs], #2 [FT], #3 [abs],
  #4 [abs], #5 [FT], 2022 thesis #52; whole-case cancer analysis #50 (preprint); multi-image
  fusion #6, #7, #12.
- "First task-specific attention pooling for multi-task MIL": SydneyMTL #51 (preprint,
  histology); CLAM's class-specific branches.
- "First use of anatomical-region structure for corpus-predominant or autoimmune gastritis":
  GSCNet #8 [FT] (fixed section triplet, section-correlation losses); Cao et al. #9 [FT]
  (image-level region × atrophy classification).
- "First label-consistency constraint in medical classification" or "first constrained MIL" —
  #34–#41, #49; Giunchiglia & Lukasiewicz 2020; Xu et al. 2018.
- "Novel view dropout / cross-view consistency" — random pseudo-bags and full-vs-masked-bag
  teacher–student consistency exist (#42–#45); ModDrop for inputs generally.

Claimable, limited to what the search covered (full text for #2, #5, #8, #9; abstract for the
other six closest papers; re-run before submission):

- First patient-level MIL evaluation on **EGID**; no other EGID gastritis results exist (all
  sources incl. citation tracking, medRxiv and CNKI).
- Identification on the released data that EGID's presence and type labels are deterministic
  functions of distribution, plus a measured coherence rate of unstructured models (no coherence
  rate in #2, #5, #8, #9 [FT]; none in the abstracts of the others).
- Joint patient-level modelling of H. pylori **and** atrophy distribution from patient labels
  only, without section-specific inputs (#8 [FT]) or per-image region labels (#9 [FT]).
- A pre-specified, cross-validated negative result: on ≈133 training patients per fold,
  task-aware attention MIL with structured head and view-subset consistency does not beat
  image-level aggregation or mean pooling. The large-cohort positive reports train on 3,555–6,388
  patients (#5 [FT], #1 [abs]), and HOPE AI's own comparison [FT] already found ABMIL/TransMIL
  below top-k pooling. Present the result as consistent with, not contrary to, that literature.
- An image-count robustness curve (25/50/75 %) for patient-level gastritis classification
  (absent in #2, #5, #8, #9 [FT]; provisional for the [abs] papers).

Wording for the manuscript: CoRe-MIL is an *application and controlled evaluation* of
established MIL, constraint and consistency techniques on EGID, not a new method family.

### 4.2 Scope boundaries

- The missing-view experiment is **random image-subset robustness**: EGID provides no per-image
  anatomical location, so anatomically structured missingness (e.g. "no corpus views") cannot be
  simulated.
- The structural constraint is a property of how EGID was annotated (distribution determines
  type and presence). It is not a universal clinical law; e.g. other cohorts may label type A
  gastritis without a distribution category. The mapping is therefore learned from training data
  and checked, not hard-coded.
- Single centre, single endoscope system, retrospective selection of high-quality images: any
  positive finding is internal validity only.
