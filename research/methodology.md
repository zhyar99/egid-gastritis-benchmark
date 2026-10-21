# Methodology: CoRe-MIL

Constraint-aware and missing-view-robust multiple-instance learning for patient-level,
multi-label gastritis classification on EGID. This document defines what the code in `src/`
implements; numbered modules match the paper's Figure 1 (`outputs/figures/fig1_architecture.pdf`).

## 1. Problem setting

Patient *i* has a bag of *Nᵢ* white-light endoscopic images Xᵢ = {x_{i,1}, …, x_{i,Nᵢ}} (EGID:
median 24, range 6–108) and four patient-level labels: H. pylori status y^HP ∈ {positive,
negative}, atrophy presence y^A ∈ {atrophic, non-atrophic}, atrophy distribution
y^D ∈ {normal, antrum-predominant, corpus-predominant, antrum-and-corpus} and gastritis type
y^T ∈ {type A, type B, normal}. Labels are assigned per patient from overall findings
(Li et al. 2026); images carry no individual labels and no anatomical-location metadata. The
learning problem is therefore weakly supervised multiple-instance learning with four bag labels.

**Verified label structure.** On all 229 patients (`outputs/data_audit/label_relationships.json`)
there exist deterministic maps g_A: D → A and g_T: D → T,

- g_A(normal) = non-atrophic, g_A(·) = atrophic otherwise;
- g_T(normal) = normal, g_T(corpus-predominant) = type A,
  g_T(antrum-predominant) = g_T(antrum-and-corpus) = type B.

The independent label set is {y^HP, y^D}. We write the maps as 0/1 matrices M_A ∈ {0,1}^{4×2},
M_T ∈ {0,1}^{4×3} with one 1 per row. They are re-estimated on each split's training patients and
the structured head is enabled only if they are exactly deterministic there (no test information).

## 2. Model

**Module A — encoder.** f_{i,k} = φ(x_{i,k}) ∈ ℝ^d, a pretrained timm backbone (EfficientNet-B0 by
default, d = 1280) applied to a black-padded square image resized to 224 px. In the frozen mode
φ is fixed and embeddings are cached once; features are z-scored with training-image statistics.
In the fine-tuning mode φ is trained (lower learning rate; optional freezing of early layers).

**Projection.** h_{i,k} = Dropout(ReLU(W₁ Dropout_in(f_{i,k}) + b₁)) ∈ ℝ^H.

**Module B — bags.** Bags are padded to K = maxᵢ Nᵢ in a batch with a mask m_{i,k} ∈ {0,1}.

**Module C — task-aware gated attention.** For each output group g ∈ G (G = {HP, D} with the
structured head, all four tasks otherwise) an independent gated-attention branch (Ilse et al.
2018) scores every instance:

  s^g_{i,k} = w_gᵀ [ tanh(V_g h_{i,k}) ⊙ σ(U_g h_{i,k}) ],
  a^g_{i,k} = exp(s^g_{i,k}) m_{i,k} / Σ_j exp(s^g_{i,j}) m_{i,j},
  z^g_i = Σ_k a^g_{i,k} h_{i,k}.

Padded positions receive exactly zero weight (scores set to −∞ before the softmax; tested for
invariance to padded content). Baselines: mean pooling z_i = (Σ_k m_{i,k} h_{i,k}) / Σ_k m_{i,k};
a single shared branch (|G| = 1, "B4").

**Module D — structured clinical head.** Logits ℓ^g_i = W_g z^g_i + b_g for g ∈ {HP, D}, with
p^HP_i = softmax(ℓ^HP_i), p^D_i = softmax(ℓ^D_i), and the dependent tasks

  p^A_i = p^D_i M_A,  p^T_i = p^D_i M_T.

These are valid distributions and the dependent *probabilities* are consistent with p^D by
construction. Per-task *argmax decisions* are not automatically coherent: e.g. p^D = (0.4, 0.2, 0.2,
0.2) predicts *normal* distribution but p^A(atrophic) = 0.6. We keep per-task argmax decisions (the
Bayes decision for each task's accuracy) and report the *coherence rate* — the fraction of patients
whose four decisions form a valid combination under (M_A, M_T). With independent heads, each task
has its own softmax and coherence is not guaranteed; an optional soft constraint is

  L_clin = ½ Σ_{t∈{A,T}} KL( p^D M_t ‖ p^t ).

**Image-level baseline (EGID protocol).** Each image is classified by four heads on the shared
feature; the patient prediction is the mean of image softmax probabilities, p^t_i = (1/Nᵢ) Σ_k
softmax(ℓ^t_{i,k}) (Li et al. 2026). Its training loss is image-level cross-entropy with the
patient's labels.

## 3. Training objective

**Supervised term.** For a view V of the bag,
L_sup(V) = (1/4) Σ_{t} ℓ_t( p^t(V), y^t ), with ℓ_t = cross-entropy (optionally class-weighted with
inverse patient frequencies of the training split, or focal with γ = 2). Missing labels (y = −1)
are masked; with EGID none are missing.

**Module E — view sampling.** Two views of each training bag per step: a teacher view V^T (full bag
by default; configurable ratio range) and a student view V^S ⊂ V^T with |V^S| = max(1, round(r·Nᵢ)),
r ~ U[0.2, 0.8]. Alternatives: Bernoulli view dropout (p = 0.5) or absolute bag sizes k ~ U{2..12}.
Without view sampling only the full bag is used.

**Module F — confidence-gated cross-view consistency.** For each output group g,

  L_cv = T² · (1/B) Σ_i 𝟙[max_c p^{g,T}_{i,c} ≥ τ] · KL( sg[softmax(ℓ^{g,T}_i / T)] ‖ softmax(ℓ^{g,S}_i / T) ),

where sg is stop-gradient (the teacher is the same network on the larger view, detached), τ is
the confidence gate (evaluated at temperature 1), T the distillation temperature, B the batch size.
Patients whose larger view is not confident contribute nothing, so the student is not pulled
towards uncertain targets.

**Total.**
L = ½ [L_sup(V^T) + L_sup(V^S)] + λ₁ L_clin + λ₂ w(e) L_cv,
w(e) = 0 for e < e₀ and min(1, (e − e₀ + 1)/R) afterwards (e₀ = 5, R = 15). Terms not applicable to
a variant are absent (e.g. L_clin only for independent heads; L_cv only with two views). The A5
vs A6 comparison isolates L_cv exactly: both use the same two views and supervised terms.

**Optimisation.** AdamW, mini-batches of 16 patients (256 images for image-level heads), early
stopping on inner-validation mean NLL (patience 25), best checkpoint restored.

## 4. Inference, uncertainty and calibration (Module G)

Prediction uses the full available bag (or the evaluated subset). Outputs per task: class
probabilities, predicted class (argmax), confidence max_c p_c and entropy −Σ p_c log p_c.
Post-hoc temperature scaling (Guo et al. 2017) fits one scalar T_g per independent output group
by minimising validation NLL of softmax(ℓ^g / T_g) (for the image-level baseline, ℓ^g := log of the
mean probability, so T = 1 leaves the EGID aggregation unchanged); dependent tasks inherit
calibration through M. Softmax probabilities are model scores, not clinically validated certainty.

## 5. Evaluation

Patient-level only. Metrics: accuracy, balanced accuracy, macro/weighted precision, recall and F1,
per-class F1/recall/precision, confusion matrix, AUROC and AUPRC (one-vs-rest macro; undefined
and reported as such when a class is absent), NLL, Brier score, top-label ECE (10 bins), coherence
rate. Missing-view robustness: for fraction f ∈ {1, 0.75, 0.5, 0.25} and draw r, the subset of
patient *i* is a deterministic function of (i, f, r), shared by all models. Statistics: see
`experiment_protocol.md` §7.

## 6. Implementation map

| Component | File |
|---|---|
| Audit, label relationships, duplicates | `src/data/audit.py` |
| Splits (paper-matched, CV, group-aware) | `src/data/splits.py` |
| Image cache / bags / view sampling | `src/data/preprocessing.py`, `src/data/dataset.py` |
| Encoders, feature cache | `src/models/encoders.py`, `src/training/feature_extraction.py` |
| MIL, attention, structured head | `src/models/core_mil.py`, `src/models/structured_head.py` |
| Image-level baseline | `src/models/baseline.py` |
| Losses | `src/training/losses.py` |
| Training, evaluation, records | `src/training/trainer.py`, `src/training/finetune.py` |
| Metrics, calibration, robustness, statistics | `src/evaluation/*.py` |
