# Literature comparison

Based on the systematic search of 2026-10-19 (`literature_search.md`). Main round: PubMed,
arXiv, IEEE Xplore, and OpenAlex as a Scopus substitute; 1,161 records, 980 unique. Second round:
Google Scholar (partial), medRxiv and CNKI. 50 studies were included from the searches. Every
reference below was verified against its DOI record, PubMed, the publisher page or the
proceedings page; `#` numbers refer to `literature_search.md` §7. Statements about the ten
closest papers carry a tag: **[FT]** = checked in the full text, **[abs]** = abstract only (paywalled,
no preprint). The full claim-by-claim matrix is in `literature_search.md` §7.7.

## Dataset and reference results

| Work | Relevance |
|---|---|
| Li Z, Fang Y, Song Y, Bai H, Li T, Wang Y, Zhang Q, Li X. EGID: a comprehensive multi-label endoscopic image dataset for gastritis classification. *Sci Data* 2026;13:1396. doi:10.1038/s41597-026-07666-6 | Dataset; image-level multitask baselines; split sizes/composition used here. |

Reported EGID test results (literature benchmark, *not* our results; 34 test patients; Li et al.
Table 4, re-checked 2026-10-19):

| Backbone | Mean acc | Mean bal. acc | Mean macro-F1 | Mean AUROC |
|---|---|---|---|---|
| ResNet18 | 72.06 | 69.21 | 69.05 | 88.74 |
| ResNet50 | 78.68 | 74.45 | 75.08 | 91.95 |
| EfficientNet-B0 | 69.85 | 62.17 | 64.70 | 88.92 |
| DenseNet121 | 83.82 | 81.74 | 82.99 | 93.60 |

DenseNet121 per task (acc / macro-F1; Li et al. Table 5): H. pylori 94.12 / 93.93; presence
82.35 / 75.48; distribution 73.53 / 76.06; type 85.29 / 86.13.

**Other uses of EGID.** None with gastritis-label results. The only citing work found (Tran & Dang,
woma, arXiv:2609.15130, 2026; #33) pools EGID images into foundation-model training.

## Closest prior work: patient-level learning on gastroscopy

| Study | Source checked | Task(s) | Patients | Aggregation | Multi-task / label structure | Trained or tested for fewer images? |
|---|---|---|---|---|---|---|
| Yi et al. 2026, PhenoMIL (*Med Image Anal*) #1 | [abs] | H. pylori (+ phenotype multi-label) | 6,388 | Attention MIL, phenotype-guided | Phenotype → diagnosis prior; extra instance annotations | not stated in abstract |
| Hu et al. 2025, HOPE AI (*BMC Med*) #2 | [FT] incl. supplement | H. pylori +/− | 6,207 | Top-k (k = 7) mean pooling + LSTM; ≥ 8 images required, ≈35 per case | no | **no** |
| Ting et al. 2025 (*IEEE CIBCB*) #3 | [abs] + first page | H. pylori | n.r. | Transformer MIL, class tokens | no | not stated |
| Jiang & Lv 2025, DwinTMIL (*IEEE ICIPCA*) #4 | [abs] | H. pylori | n.r. | Swin + MIL | no | not stated |
| Chen Z et al. 2026, EndoStrat (*Front Med*) #5 | [FT] | Normal / atrophy-IM / LGIN / HGIN-EGC triage | 8,829 (3,555 training) | **Gated attention MIL (Ilse)** on ResNet-50 features, as in our A1; ≥ 20 images required (median 25) | One label per patient, two-stage cost-sensitive | **no** |
| Dai J et al. 2026 (*Surg Endosc*) #6 | [abs] | H. pylori | 5,865 | Multi-region, multi-image CNN | no | single- vs multi-image compared |
| Dai D et al. 2025 (*J Gastroenterol Hepatol*) #7 | [abs] | H. pylori status (3-class) | 802 | Voting over stomach sites | no | single- vs multi-site compared |
| Jhang et al. 2024, GSCNet (*IEEE OJEMB*) #8 | [FT] | Corpus-predominant gastritis index (binary) | not quotable (Table I inconsistent) | Fixed triplet: one antrum, one body, one cardia image | Section-correlation losses from the CGI definition; single output | **no**; input cannot be incomplete |
| Cao Y et al. 2025 (*J Imaging*) #9 | [FT] | Image-level anatomical region + atrophy (AIG motivation) | 298 | Images grouped by SIFT registration, then **per-image** classification | Two image-level tasks with image-level labels; **no patient-level AIG diagnosis reported** | **no** |
| Chen S et al. 2025 (*Endoscopy*) #10 | [abs] | AIG vs HpAG vs non-atrophic | 1,070 | Multi-site feature fusion | no | not stated |
| Özden & Omurca 2026 (*BMC Gastroenterol*) #11 | H. pylori | 172 | Mean of image predictions | no | not stated; 5-fold CV |
| Li et al. 2026, EGID reference (*Sci Data*) | 4 EGID tasks | 229 | Mean of image softmax | 4 independent heads | no |
| **CoRe-MIL (this work)** | H. pylori + distribution (→ presence, type) | 229 | Task-specific gated attention; mean-pooling variant | Structured head + L_clin from mapping verified per split | Random 20–80 % image subsets in training; tested at 25/50/75 % |

n.r. = not reported in the abstract.

Also relevant (second-round search): GutCore (Kim et al., medRxiv 2026, #50) aggregates
whole-case gastroscopy image sets for patient-level cancer endpoints at foundation-model scale.
A 2022 Zhejiang University master's thesis (#52, CNKI, grey literature) already used
multi-instance learning for H. pylori from gastroscopy images and video.

Two full-text findings bear directly on our negative result:
- EndoStrat (#5) [FT] uses the same gated-attention pooling as our A1 and beats a per-image
  classifier. Its baseline pools by **max**, ours by **mean**, and it trains on 3,555 patients
  against our ≈133 per fold.
- In HOPE AI's own comparison (#2, Suppl. Table 1) [FT], attention MIL (ABMIL, AUC 87.6) and
  TransMIL (87.8) scored *below* their simpler top-k pooling baseline (92.1) on 6,207 patients.

So the finding that attention pooling does not beat simpler aggregation is not unique to our
small cohort.

Further endoscopic deep-learning context (image-, video- or extent-level, not patient-level MIL):
H. pylori status (Shichijo 2019 #14; Li G 2025 #15; Li YD 2023 video #16; Li F 2026 #17, which
shows H. pylori detection is harder in atrophic mucosa); atrophy extent/staging (Tao 2024
Kimura-Takemoto #18; Zhao 2023 OLGA #19; Kodaka 2022 #20); AIG (Song 2026 #21; Cao 2026
multimodal #22; Yang F 2026 #23); joint or multi-label findings (Lin 2021 AG+IM #24; Liu 2024
Kyoto score #25); video consistency (Adaptify, Xiong 2024 #26); public atrophy/IM images
(Yang J 2023 #27); meta-analyses (Jiang 2025 #28; Parkash 2024 #29; Shi 2023 #30). Multi-task
gastritis MIL exists on histology (GastritisMIL, Xia 2025 #31; SydneyMTL, Jeong 2026 #51, which uses
**task-specific attention pooling**, i.e. one attention branch per task, as our Module C does),
not on endoscopy.

## Methodological building blocks

| Area | Work | How CoRe-MIL relates |
|---|---|---|
| Attention MIL | Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. *Proc 35th ICML*, PMLR 2018;80:2127–2136. | Gated attention pooling (Module C) is taken from here; we use one branch per output group. |
| Clinical MIL | Campanella G, Hanna MG, Geneslaw L, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nat Med* 2019;25(8):1301–1309. doi:10.1038/s41591-019-0508-1 | Patient/slide-level weak supervision at scale. |
| Multi-branch attention MIL | Lu MY, Williamson DFK, Chen TY, et al. Data-efficient and weakly supervised computational pathology on whole-slide images. *Nat Biomed Eng* 2021;5(6):555–570. doi:10.1038/s41551-020-00682-w | Class-specific attention branches. Task-specific attention pooling for multi-task MIL: SydneyMTL (Jeong WC et al., medRxiv 2026, doi:10.64898/2026.02.17.26346304; #51), on histology. Ours applies the same idea to endoscopy bags. |
| Transformer MIL | Shao Z, Bian H, Chen Y, et al. TransMIL: transformer based correlated multiple instance learning for whole slide image classification. *Adv Neural Inf Process Syst* 2021;34:2136–2147. | Alternative aggregator, not implemented (small cohort); transformer MIL for H. pylori: Ting 2025 #3. |
| Hierarchical / logical consistency | Giunchiglia E, Lukasiewicz T. Coherent hierarchical multi-label classification networks. *Adv Neural Inf Process Syst* 2020;33:9662–9673. Xu J, Zhang Z, Friedman T, Liang Y, Van den Broeck G. A semantic loss function for deep learning with symbolic knowledge. *Proc 35th ICML*, PMLR 2018;80:5502–5511. | Hard (structured head) and soft (L_clin) consistency with label relations; ours are verified from data per split. |
| Medical hierarchical multi-label | Chen H et al. *Med Image Anal* 2020 #34; Ju L et al. *IEEE TMI* 2024 #35; Yang Z et al. *IEEE TMI* 2026 #36; Wang T et al. C²Net, *Med Biol Eng Comput* 2026 #37; Xu Y et al. *IEEE JBHI* 2022 #38; Strong J et al. arXiv 2026 #40 | Label-consistency in medical imaging is established (CXR, fundus); C²Net explicitly penalises "normal + disease" conflicts, the analogue of our "non-atrophic + type B". |
| Constrained MIL | Lee C et al. TaxoMIL, arXiv:2606.31100 (2026) #39; Zhang P et al. EchoHMIL, *BMC Med Imaging* 2026 #41; Shiku K et al. ordinal MIL, *WACV* 2025 #49 | MIL with taxonomy/hierarchical/ordinal output structure exists (pathology, echo, colonoscopy). |
| Missing modality / view dropout | Neverova N, Wolf C, Taylor G, Nebout F. ModDrop: adaptive multi-modal gesture recognition. *IEEE Trans Pattern Anal Mach Intell* 2016;38(8):1692–1706. doi:10.1109/TPAMI.2015.2461544; Zhao C et al. Diffmv, arXiv:2505.11802 (2025) #47 | Training-time dropping of inputs; here, of instances within a bag. |
| Instance subsampling / masking in MIL | Zhang H et al. DTFD-MIL, *CVPR* 2022 #42; Tang W et al. MHIM-MIL, *ICCV* 2023 #43; Zhang Y et al. ACMIL, *ECCV* 2024 #44; Liu P et al. PseMix, *IEEE TMI* 2024 #45; Dasdelen MF et al. arXiv 2026 #46 | **Random instance subsets as training views (pseudo-bags) and teacher–student consistency between full and masked bags (MHIM-MIL) are established in WSI MIL.** Our view dropout + detached full-view teacher is an instance of this family, transferred to small endoscopy bags. |
| Consistency / self-distillation | Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531 (2015); Sohn K, Berthelot D, Carlini N, et al. FixMatch: simplifying semi-supervised learning with consistency and confidence. *Adv Neural Inf Process Syst* 2020;33:596–608. | Temperature-scaled KL and confidence-thresholded targets (Module F). |
| Calibration | Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *Proc 34th ICML*, PMLR 2017;70:1321–1330. | Temperature scaling, ECE (Module G). |
| Class imbalance | Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. *Proc IEEE ICCV* 2017:2999–3007. doi:10.1109/ICCV.2017.324 | Focal-loss variant X3. |
| Endoscopic H. pylori CNNs (early) | Shichijo S, Nomura S, Aoyama K, et al. Application of convolutional neural networks in the diagnosis of Helicobacter pylori infection based on endoscopic images. *EBioMedicine* 2017;25:106–111. doi:10.1016/j.ebiom.2017.10.014; Itoh T, Kawahira H, Nakashima H, Yata N. Deep learning analyzes Helicobacter pylori infection by upper gastrointestinal endoscopy images. *Endosc Int Open* 2018;6(2):E139–E144. doi:10.1055/s-0043-120830; Zheng W et al. *Clin Transl Gastroenterol* 2019 #12 | Image-level CNNs; Zheng 2019 already showed per-patient multi-image aggregation beats single images. |

## Positioning after the search

What the search rules out as a novelty claim:

1. **Patient-level / attention MIL on gastroscopy** is not new. It exists for H. pylori (#1 [abs],
   #2 [FT], #3 [abs], #4 [abs], and the 2022 thesis #52), at up to 30× our cohort size and with
   prospective validation (#2 [FT]). It also exists for precancerous triage including atrophy,
   with the same gated-attention pooling we use (#5 [FT]), and for whole-case cancer analysis
   (#50, preprint). Multi-image fusion beats single images in #6 [abs], #7 [abs] and #12. This
   agrees with our finding that patient-level aggregation helps; our attention model did not
   beat simple aggregation on EGID.
2. **Task-specific attention pooling for multi-task MIL** is not new. SydneyMTL (#51, preprint,
   histology) uses one attention pooling per task; CLAM uses class-specific branches.
3. **Using gastric-region structure for corpus-predominant / autoimmune gastritis** is not new as
   an idea. GSCNet (#8 [FT]) encodes the antrum-vs-corpus definition of the corpus-predominant
   index in section-correlation losses, but needs a fixed triplet of section-specific images.
   Cao et al. (#9 [FT]) classify each image's region and atrophy with image-level labels and do
   not report a patient-level diagnosis. Neither approach is possible on EGID, which has no
   per-image region or atrophy labels.
4. **Label-consistency constraints** (#34–#40, Giunchiglia 2020, Xu 2018) and **constrained MIL**
   (#39, #41, #49) are established methods.
5. **Instance subsampling / masking with full-bag teachers** is established in WSI MIL (#42–#45).

What remains, as far as the search can show (EGID-specific). Tags give the evidence level for
the closest papers:

- The first patient-level MIL results on **EGID** beyond the descriptor's image-level baselines.
  No other EGID results exist as of 2026-10-19; the search covered PubMed, arXiv, IEEE, OpenAlex,
  medRxiv, CNKI, citation tracking, and woma read in full text.
- The observation, verified on the released annotations, that EGID's "independent" presence and
  type labels are deterministic functions of distribution, plus a measurement of how often
  unstructured models output incoherent combinations. No included gastritis study reports a
  coherence rate: [FT] for #2, #5, #8, #9; [abs] for #1, #3, #4, #6, #7, #10.
- A **joint** H. pylori + atrophy-distribution patient-level model trained from patient labels
  only. Prior patient-level MIL work targets H. pylori (with phenotype labels in #1 [abs]) or a
  single triage label (#5 [FT]). Distribution/AIG work uses section-specific inputs (#8 [FT]) or
  image-level region labels (#9 [FT]); #10 [abs] fuses fixed sites.
- A controlled comparison under a pre-specified protocol on a small cohort (≈133 training patients
  per fold). It shows that task-specific attention MIL, structured heads and view-subset
  consistency **do not** beat image-level aggregation or mean pooling. This is consistent with
  HOPE AI's own finding that ABMIL/TransMIL scored below top-k pooling (#2 [FT]), and it qualifies
  the large-cohort attention-MIL results (#1 [abs], #5 [FT]).
- An explicit image-count robustness curve (25/50/75 % of images) for gastritis classification.
  None of #2, #5, #8, #9 trains or tests with fewer images [FT]; #6 and #7 compare only one
  image/site against all [abs]; #1, #3, #4, #10 do not mention it [abs]. The claim stays
  provisional until those six full texts are read (`literature_search.md` §8 open items).

These are contributions of evaluation and problem formulation on one public dataset, not of new
architecture components.
