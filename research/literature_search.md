# Systematic literature search (PRISMA-style log)

Search run: **2026-10-19** (all databases queried that day; supplementary round the same day:
Google Scholar, medRxiv, CNKI, full-text checks, §3a/§7.7). Publication window: **2018-01-01 to
2026-12-31**. Screening and data extraction: single reviewer (AI agent session working for the
author), no second reviewer. Full texts were read for the four open-access closest papers
(§7.7); see §8 for what this does and does not support and for the items still open.

## 1. Questions the search had to answer

| ID | Question | Why it matters for CoRe-MIL |
|---|---|---|
| S1 | Has multiple-instance / patient-level learning been applied to gastroscopy for H. pylori, atrophy or gastritis type? | Gap G1 and the claim that EGID was only handled by label copying |
| S2 | Which deep-learning work exists on H. pylori, atrophic gastritis, atrophy *distribution* / extent, autoimmune (type A) gastritis? | Clinical positioning; prior use of anatomical-region structure |
| S3 | Has EGID (Li et al., *Sci Data* 2026) been used by anyone else? | Whether our EGID results are the first beyond the data descriptor |
| S4 | Hierarchical / structured label consistency in medical multi-label or multi-task classification | Gap G2, structured head and L_clin |
| S5 | Missing-view / instance-dropout / bag-subsampling robustness in MIL | Gap G3, view dropout and cross-view consistency |

## 2. Sources and how they were accessed

| Source | Access | Notes |
|---|---|---|
| PubMed | NCBI E-utilities `esearch`/`esummary`/`efetch` (no API key) | Field `[tiab]`, date `2018:2026[dp]`; all PMIDs and abstracts retrieved |
| arXiv | arXiv API (`export.arxiv.org/api/query`) | Field `abs:`, `submittedDate:[201801010000 TO 202612312359]`; all records + abstracts retrieved |
| IEEE Xplore | Public IEEE Xplore search (the site's own search endpoint, called from an anonymous browser session); filter 2018–2026 | Metadata + snippet abstracts only (no subscription); full records for the first 100 hits per query |
| Scopus | **Not accessible**: Scopus web UI redirects to Elsevier sign-in and the Scopus API rejects requests without a key | Replaced by **OpenAlex** (`title_and_abstract.search`, 2018–2026), whose source list covers Scopus-indexed journals and proceedings. Re-running the queries in Scopus itself remains an open item (§8). |
| Google Scholar | Public web search in a browser session | One query answered (§3a); the next query returned Google's "unusual traffic" CAPTCHA page. CAPTCHAs are not bypassed, so Scholar coverage stops there. |
| medRxiv | Europe PMC preprint index restricted to medRxiv (`SRC:PPR AND PUBLISHER:"medRxiv"`) | medRxiv's own site has no usable keyword API; Europe PMC indexes all medRxiv preprints incl. full text, so hits include full-text matches (noisy) |
| CNKI | CNKI international site (oversea.cnki.net), subject search (`SU`), Chinese terms | Five queries ran (§3a). Opening a record page showed a slider verification, so only one record page was read and no further CNKI searching was done. |
| Citation tracking for EGID | Crossref (`is-referenced-by-count`), OpenAlex (`cited_by_count`), Semantic Scholar citations endpoint, Europe PMC full-text search | §4 |
| Bibliographic verification | Crossref works API (DOI records), PubMed `esummary`, publisher/proceedings landing pages (PMLR, NeurIPS proceedings), arXiv API, nature.com article/tables | §7 |

## 3. Queries and hit counts

Hit count = total reported by the database for the exact string below (with the date filter).
"Retrieved" = records downloaded for screening (all of them, except IEEE where ≤100 per query).

### PubMed (date limit appended to every query: `AND 2018:2026[dp]`)

Shorthands, expanded literally in the submitted strings:
`DL` = `("deep learning"[tiab] OR "convolutional neural network*"[tiab] OR "artificial intelligence"[tiab] OR "machine learning"[tiab] OR "neural network*"[tiab] OR "computer-aided"[tiab] OR transformer*[tiab])`;
`UGI` = `(endoscop*[tiab] OR gastroscop*[tiab] OR esophagogastroduodenoscop*[tiab] OR "upper gastrointestinal"[tiab])`.

| ID | Q | Query | Hits |
|---|---|---|---|
| P1 | S1 | `("multiple instance learning"[tiab] OR "multiple-instance learning"[tiab] OR "multi-instance learning"[tiab]) AND (endoscop*[tiab] OR gastroscop*[tiab] OR esophagogastroduodenoscop*[tiab] OR colonoscop*[tiab] OR "capsule endoscopy"[tiab])` | 18 |
| P2 | S1 | `("patient-level"[tiab] OR "case-level"[tiab] OR "examination-level"[tiab] OR "patient level"[tiab] OR "per-patient"[tiab] OR "weakly supervised"[tiab] OR "weakly-supervised"[tiab] OR "multiple instance"[tiab]) AND DL AND (gastritis[tiab] OR atroph*[tiab] OR pylori[tiab] OR "intestinal metaplasia"[tiab] OR gastric[tiab]) AND UGI` | 19 |
| P3 | S2 | `("Helicobacter pylori"[tiab] OR "H. pylori"[tiab]) AND DL AND UGI` | 119 |
| P4 | S2 | `("atrophic gastritis"[tiab] OR "gastric atrophy"[tiab] OR "chronic atrophic"[tiab] OR "intestinal metaplasia"[tiab] OR "precancerous"[tiab] OR "gastric premalignant"[tiab]) AND DL AND UGI` | 267 |
| P5 | S2 | `("Kimura-Takemoto"[tiab] OR "Kimura Takemoto"[tiab] OR "atrophic border"[tiab] OR "extent of atrophy"[tiab] OR "atrophy extent"[tiab] OR "corpus-predominant"[tiab] OR "corpus predominant"[tiab] OR OLGA[tiab] OR OLGIM[tiab] OR "distribution of atrophy"[tiab] OR "atrophy distribution"[tiab] OR "Kyoto classification"[tiab]) AND DL` | 26 |
| P6 | S2 | `("autoimmune gastritis"[tiab] OR "type A gastritis"[tiab] OR "autoimmune atrophic gastritis"[tiab] OR "corpus atrophy"[tiab] OR "pernicious anemia"[tiab]) AND DL` | 18 |
| P7 | S3 | `(EGID[tiab] AND (gastritis[tiab] OR endoscop*[tiab])) OR "10.1038/s41597-026-07666-6"[aid]` | 94 (93 are "eosinophilic gastrointestinal disease"; 1 = EGID descriptor) |
| P8 | S4 | `("hierarchical multi-label"[tiab] OR "hierarchical multilabel"[tiab] OR "label hierarchy"[tiab] OR "hierarchical classification"[tiab] OR "label consistency"[tiab] OR "label dependenc*"[tiab] OR "logical constraint*"[tiab] OR "semantic loss"[tiab] OR "taxonomy-aware"[tiab] OR "hierarchy-aware"[tiab]) AND DL AND (medical[tiab] OR clinical[tiab] OR endoscop*[tiab] OR radiolog*[tiab] OR patholog*[tiab] OR dermatolog*[tiab] OR "chest x-ray"[tiab] OR fundus[tiab])` | 107 |
| P9 | S5 | `("multiple instance"[tiab] OR "multi-instance"[tiab] OR "multi-view"[tiab] OR "multiview"[tiab]) AND ("missing view*"[tiab] OR "missing instance*"[tiab] OR "instance dropout"[tiab] OR "view dropout"[tiab] OR "incomplete view*"[tiab] OR "missing modalit*"[tiab] OR "bag size"[tiab] OR "number of images"[tiab] OR "instance sampling"[tiab] OR "random subset*"[tiab]) AND DL` | 16 |
| | | **PubMed total** | **684** |

### arXiv (date limit appended: `AND submittedDate:[201801010000 TO 202612312359]`)

| ID | Q | Query | Hits |
|---|---|---|---|
| A1 | S1 | `(abs:"multiple instance" OR abs:"multi-instance") AND (abs:endoscopy OR abs:endoscopic OR abs:gastroscopy OR abs:colonoscopy)` | 9 |
| A2 | S2 | `(abs:gastritis OR abs:"gastric atrophy" OR abs:pylori OR abs:"intestinal metaplasia")` | 27 |
| A3 | S3 | `abs:EGID` | 1 (unrelated: astronomy "EGIDE") |
| A4 | S4 | `(abs:"hierarchical multi-label" OR abs:"label hierarchy" OR abs:"logical constraints" OR abs:"semantic loss" OR abs:"hierarchy-aware") AND (abs:medical OR abs:clinical)` | 35 |
| A5 | S5 | `(abs:"multiple instance" OR abs:"multi-instance") AND (abs:"missing" OR abs:"instance dropout" OR abs:"bag size" OR abs:"incomplete")` | 60 |
| A6 | S5 | `(abs:"missing view" OR abs:"missing views" OR abs:"view dropout" OR abs:"incomplete views") AND (abs:medical OR abs:clinical OR abs:mammograph* OR abs:patient)` | 10 |
| | | **arXiv total** | **142** |

### IEEE Xplore (All Metadata, year filter 2018–2026)

| ID | Q | Query | Hits |
|---|---|---|---|
| I1 | S1 | `("multiple instance learning" OR "multiple-instance learning" OR "multi-instance learning") AND (gastroscopy OR endoscopy OR endoscopic)` | 12 |
| I2 | S1/S2 | `(gastritis OR "gastric atrophy" OR "Helicobacter pylori" OR "H. pylori" OR "intestinal metaplasia") AND (endoscopy OR endoscopic OR gastroscopy) AND ("deep learning" OR "convolutional neural network" OR "neural network" OR transformer)` | 65 |
| I3 | S2 | `("autoimmune gastritis" OR "corpus-predominant" OR "Kimura-Takemoto" OR "atrophy distribution" OR "distribution of atrophy" OR "OLGA staging" OR "atrophic gastritis") AND ("deep learning" OR "neural network" OR "machine learning")` | 22 |
| I4 | S3 | `"EGID" AND gastritis` | 0 |
| I5 | S4 | `("hierarchical multi-label" OR "label hierarchy" OR "label consistency" OR "logical constraints" OR "semantic loss" OR "hierarchy-aware") AND (medical OR clinical OR endoscopy OR fundus OR "chest x-ray")` | 89 |
| I6 | S5 | `("multiple instance" OR "multi-instance" OR "multi-view") AND ("missing view" OR "missing views" OR "instance dropout" OR "view dropout" OR "incomplete views" OR "bag size") AND (medical OR clinical OR patient)` | 9 |
| | | **IEEE total** | **197** |

A first version of I3 that contained the bare term `OLGA` returned 3,515 hits (matches unrelated
uses of the word) and was replaced by the string above before screening; it is not counted.

### OpenAlex (Scopus substitute; `title_and_abstract.search`, publication date 2018-01-01..2026-12-31)

| ID | Q | Query | Hits |
|---|---|---|---|
| O1 | S1 | `("multiple instance learning" OR "multi-instance learning") AND (gastroscopy OR endoscopy OR endoscopic)` | 47 |
| O2 | S1 | `("patient-level" OR "multiple images" OR "multi-image" OR "multisite" OR "multi-site") AND (gastritis OR atrophy OR pylori) AND (endoscopy OR endoscopic OR gastroscopy) AND ("deep learning" OR "neural network")` | 5 |
| O3 | S2 | `("autoimmune gastritis" OR "corpus-predominant" OR "Kimura-Takemoto" OR "atrophy distribution" OR "distribution of atrophy") AND ("deep learning" OR "neural network" OR "artificial intelligence")` | 37 |
| O4 | S3 | `"EGID" AND gastritis AND (dataset OR "deep learning")` | 6 (EGID descriptor, its two Figshare records, 3 eosinophilic-disease papers) |
| O5 | S5 | `("multiple instance learning" OR "multi-instance learning") AND ("instance dropout" OR "pseudo-bag" OR "pseudo bag" OR "bag size" OR "missing instances" OR "instance masking" OR "random instance")` | 43 |
| | | **OpenAlex total** | **138** |

## 3a. Supplementary sources (2026-10-19, second round)

### Google Scholar (custom range 2018–2026, `hl=en`)

| ID | Q | Query | Hits | Screened |
|---|---|---|---|---|
| G1 | S1 | `"multiple instance learning" gastroscopy "Helicobacter pylori"` | ≈70 ("About 70 results") | first 10 |
| G2 | S1/S2 | `"multiple instance learning" (gastroscopy OR endoscopy OR endoscopic) (gastritis OR "Helicobacter pylori" OR atrophy)` | — blocked by CAPTCHA, not run | — |

G1 top 10: HOPE AI (#2), DwinTMIL (#4), Ting et al. (#3), GastritisMIL (#31) — all already
included; SydneyMTL (medRxiv; included below as #51); SamMIL (IEEE CAIBDA 2025, H. pylori
grading on **pathology** — excluded, histology) and QLabelMIL (MICCAI workshop, gastric
histopathology — excluded); two arXiv reviews of computer vision in gastric cancer (2020;
excluded, review) and a pathology dataset paper (excluded). No new endoscopic study.

### medRxiv via Europe PMC (every query `AND SRC:PPR AND PUBLISHER:"medRxiv" AND FIRST_PDATE:[2018-01-01 TO 2026-12-31]`)

| ID | Q | Query | Hits |
|---|---|---|---|
| M1 | S1 | `("multiple instance learning" OR "multi-instance learning" OR "patient-level") AND (gastroscopy OR endoscopy OR endoscopic OR esophagogastroduodenoscopy)` | 20 |
| M2 | S2 | `(gastritis OR "gastric atrophy" OR "Helicobacter pylori" OR "intestinal metaplasia" OR "autoimmune gastritis") AND ("deep learning" OR "artificial intelligence" OR "neural network")` | 9 |
| M3 | S3 | `"EGID" AND (gastritis OR endoscopic)` | 0 |
| M4 | S5 | `("multiple instance learning" OR "multi-view") AND ("missing" OR "instance dropout" OR "pseudo-bag" OR "number of images")` | 11 |
| | | **medRxiv total** | **40** |

Included: GutCore (#50, M1) and SydneyMTL (#51, M2). The other 38 were full-text matches on
unrelated topics (health services, IBD, Barrett's histology, COVID-19, EHR NLP, oncology genomics)
or histology-only H. pylori detection (2020) and were excluded on title.

### CNKI (oversea.cnki.net, subject field, no date filter in URL; years screened manually)

| ID | Q | Query (Chinese; English gloss) | Hits |
|---|---|---|---|
| C1 | S2 | `胃炎 深度学习 内镜` (gastritis, deep learning, endoscopy) | 38 (12 journal articles, 26 theses) |
| C2 | S1 | `多示例学习 胃镜` (multiple-instance learning, gastroscopy) | 1 |
| C3 | S1 | `多示例 胃炎` (multiple instance, gastritis) | 0 |
| C4 | S2 | `幽门螺杆菌 深度学习 胃镜` (H. pylori, deep learning, gastroscopy) | 2 |
| C5 | S2 | `萎缩性胃炎 人工智能` (atrophic gastritis, artificial intelligence) | 37 |
| C6 | S2 | `自身免疫性胃炎 人工智能` (autoimmune gastritis, artificial intelligence) | 1 |
| | | **CNKI total** | **79** |

First 20 records of C1 and C5, and all records of C2, C4 and C6, were screened on title (the
C1 and C5 overflow beyond the first page was not screened). Included: one Zhejiang University
master's thesis on MIL for H. pylori from gastroscopy images and video (#52, C2), which is grey
literature and is cited only for completeness. Most CNKI hits were image-level CAG/IM classifiers
in master's/doctoral theses, tongue-image or TCM studies, and reviews. None used patient-level
multi-task structure or EGID. Titles only. CNKI abstracts were not read except for #52.

## 4. EGID citation tracking (S3)

| Source | Result for doi:10.1038/s41597-026-07666-6 |
|---|---|
| Crossref `is-referenced-by-count` | 0 |
| OpenAlex `cited_by_count` | 0 |
| Semantic Scholar citations | 1: Tran T, Dang L. *woma: a real-time foundation model and its fine-tuned models for endoscopy.* arXiv:2609.15130 (2026) |
| Europe PMC full text (`"s41597-026-07666-6"`, `"EGID dataset"`, `"Endoscopic Gastritis Image Dataset"`) | only the descriptor itself |

The woma preprint was read in full text for its EGID usage: EGID is one of eleven image sources
pooled for self-supervised pre-training and fine-tuning of gastroscopy station/finding models
(reference [55] there). It reports **no** results on the EGID gastritis labels. Conclusion for S3:
as of 2026-10-19 no study other than the descriptor reports patient-level classification results
on EGID.

## 5. Eligibility criteria

Included if all of: (a) 2018–2026 (pre-2018 foundational method papers already cited are kept but
were not searched for); (b) peer-reviewed article, proceedings paper or arXiv preprint with an
abstract; (c) relevant to S1–S5 as follows:

- S1/S2: deep learning on **endoscopic images or videos of the stomach** for H. pylori, atrophy
  (presence, grade, extent, distribution, Kimura-Takemoto/OLGA), gastritis type or autoimmune
  gastritis; priority to studies that aggregate several images per patient, use multi-task or
  region-aware outputs, or are systematic reviews/meta-analyses;
- S3: any use of EGID;
- S4: methods that enforce or exploit relations between labels/tasks in medical classification;
- S5: methods that subsample, drop, mask or mix instances/views in MIL, or train for missing views.

Excluded: histopathology-only gastritis work (kept one MIL example for contrast), non-gastric
endoscopy except as MIL-method precedent, serum/questionnaire/clinical-variable models, Raman/
hyperspectral/tongue imaging, cancer detection/staging without gastritis endpoints, narrative
reviews without new data (except where cited for context), conference abstracts, letters/replies,
records whose abstract could not be obtained.

## 6. Screening and flow (PRISMA 2020 items 16a)

| Stage | n |
|---|---|
| Records identified, 4 databases (PubMed 684, arXiv 142, IEEE Xplore 197, OpenAlex 138) | 1,161 |
| + citation tracking (§4) / targeted look-ups of seed and known MIL-augmentation references (DTFD-MIL) | 1 / 1 |
| + supplementary round (§3a): Google Scholar (first 10 of ≈70), medRxiv 40, CNKI 79 | 129 screened on title (not de-duplicated against the main set; Scholar's 10 contained 4 already-included studies) |
| Duplicates removed (DOI, then normalised title) | 181 |
| Unique records screened on title | 980 |
| Automated keyword pass over abstracts (PubMed P1–P6 gastric-endoscopy records) | 276 screened, 83 flagged; titles + matched terms of all 83 reviewed |
| Abstracts read in full for eligibility | 54 (39 PubMed, 7 arXiv, 7 IEEE Xplore, 1 OpenAlex) |
| Studies listed in §7 | 52 (50 found by the searches + EGID descriptor + ModDrop seed); #42–#45 and #49 included on title and bibliographic record only (well-known MIL-augmentation / endoscopy-MIL papers), all others on abstract |
| Full text read | 5: woma (for its use of EGID) and the four open-access closest papers #2, #5, #8, #9 (§7.7); #2 including its supplementary files |

The keyword pass (PubMed, P1–P6) searched title+abstract for: multiple/multi-instance,
patient/case/examination level, per patient, aggregat\*, multi-task, multi-image, multiple images,
multi-region, multisite, hierarch\*, consisten\*, attention pool\*, bag, video level, extent,
distribution, Kimura, OLGA, corpus-predominant, incomplete, missing. Of 384 unique PubMed records
in P1–P6, 276 concerned gastric endoscopy and 83 matched. The rest of the 276 were screened on
title only: they are image- or frame-level classifiers, detection/segmentation, quality control,
or reviews, and are represented here by meta-analyses rather than individually.

Main exclusion reasons at abstract level (not exhaustively counted): EGID acronym = eosinophilic
gastrointestinal disease (all 93 non-descriptor P7 hits); histopathology slides; colon/IBD/
larynx/ENT MIL; multimodal EHR/clinical-variable models; incomplete multi-view *clustering*;
LLM/NLP coding hierarchies; abstract unavailable. One record falls in the last category:
Lai Q, Vong C-M, Wong P-K, Wang S-T, Yan T, Choi I-C, Yu H-H. Multi-scale multi-instance
multi-feature joint learning broad network (M3JLBN) for gastric intestinal metaplasia subtype
classification. *Knowl-Based Syst* 2022;249:108960. doi:10.1016/j.knosys.2022.108960 (record
verified on Crossref). Re-assessed on 2026-10-19: Crossref, OpenAlex and Semantic Scholar carry
no abstract, the article is not open access, ScienceDirect refused the anonymous browser session,
and the paper is not in PubMed or Europe PMC. From the title it is a multi-instance model for
intestinal-metaplasia *subtypes*. Subtyping (complete vs incomplete IM) is normally done on
histology, but the imaging modality could not be established. **Status: not assessable, so
excluded; no claim in this repository depends on it.** Reading it needs institutional access.

## 7. Included studies — verified reference list

Every row below was checked against the DOI record (Crossref) and, for PubMed-indexed papers,
against PubMed; proceedings without DOIs were checked on PMLR / NeurIPS proceedings pages, arXiv
records on the arXiv API. "Design" is taken from the abstract unless marked otherwise.

### 7.1 Patient-level / multi-image learning on gastroscopy (S1) — closest prior work

| # | Reference | Design (from abstract) | Relation to CoRe-MIL |
|---|---|---|---|
| 1 | Yi Z, Xu J, Hu Y, et al. Learning from limited phenotype-level annotations for promoting multiple instance learning in endoscopic Helicobacter pylori infection diagnosis. *Med Image Anal* 2026;114:104267. doi:10.1016/j.media.2026.104267 | PhenoMIL: phenotype-level semi-supervised pre-training + bag-level MIL with phenotype-guided attention; 303,910 images / 6,388 patients, multicentre | **Patient-level attention MIL for H. pylori on gastroscopy exists**, at 30× our scale, with extra phenotype labels |
| 2 | Hu Y, Xu J, Huang L, et al. Artificial intelligence–assisted endoscopic diagnosis system for diagnosing Helicobacter pylori infection: a multicenter study. *BMC Med* 2025;23:540. doi:10.1186/s12916-025-04379-2 | HOPE AI: MIL + LSTM over a patient's images; 308,887 images, 197 videos, 6,207 patients, prospective external validation | Patient-level MIL for H. pylori, clinically validated |
| 3 | Ting W-T, Tsai Y-C, Huang C-R, Cheng H-C, Sheu B-S. Helicobacter pylori infection diagnosis from endoscopic images via multi-class token based multiple instance learning. *2025 IEEE Conf Comput Intell Bioinform Comput Biol (CIBCB)*; 2025:1–8. doi:10.1109/CIBCB66090.2025.11177067 | Transformer MIL with learnable positive/negative class tokens; patient-level labels only | Patient-level transformer MIL for H. pylori |
| 4 | Jiang S, Lv Z. DwinTMIL: an innovative approach for Helicobacter pylori infection classification. *2025 IEEE 3rd Int Conf Image Process Comput Appl (ICIPCA)*; 2025:733–738. doi:10.1109/ICIPCA65645.2025.11138896 | Dynamic-window Swin + MIL on gastroscopy images | MIL for H. pylori |
| 5 | Chen Z, Ma H, Yang C, et al. Endostrat: a multicenter study of AI-assisted two-level risk stratification for gastric precancerous and early neoplastic lesion screening. *Front Med* 2026;13:1918150. doi:10.3389/fmed.2026.1918150 | Patient-level weak supervision over ~220,000 images / 8,829 patients; compared with image-level classifier and a four-class MIL classifier; classes include atrophy/IM | Patient-level weakly supervised triage incl. atrophy; image-level vs MIL comparison |
| 6 | Dai J, Li Z, Zhang X, et al. Multi-region and multi-image convolutional neural network model for detecting gastric Helicobacter pylori infection. *Surg Endosc* 2026;40(2):1526–1535. doi:10.1007/s00464-025-12423-8 | Multi-region, multi-image CNN vs single-image CNN; 5,169 cases + 696 external | Multi-image fusion beats single-image, largest gain on external data |
| 7 | Dai D, Quan X, Zheng Y, et al. A multisite fusion-based deep convolutional neural network for classification of Helicobacter pylori infection status using endoscopic images: a multicenter study. *J Gastroenterol Hepatol* 2025;40(9):2240–2247. doi:10.1111/jgh.70004 | Voting-based fusion over stomach sites; 676 subjects, 8 centres | Site-wise fusion for 3-class H. pylori status |
| 8 | Jhang J-Y, Tsai Y-C, Hsu T-C, et al. Gastric section correlation network for gastric precancerous lesion diagnosis. *IEEE Open J Eng Med Biol* 2024;5:434–442. doi:10.1109/OJEMB.2023.3277219 | GSCNet: images of antrum, body, cardia; section-correlation losses encoding medical prior; target = **corpus-predominant gastritis index** | **Closest to our distribution task and to prior-knowledge losses**; needs section labels per image, EGID has none |
| 9 | Cao Y, Zhao Y, Jin X, et al. A multitask network for the diagnosis of autoimmune gastritis. *J Imaging* 2025;11(5):154. doi:10.3390/jimaging11050154 | **Full text:** a patient's images are grouped by SIFT registration + hierarchical clustering, then RepLKNet classifies each image's anatomical region (fundus/body/antrum) and atrophy (yes/no); trained on 5,981 images with **image-level** region and atrophy labels; 298 patients (96 AIG, 102 HP+ AG, 100 HP− CAG), split 201/49/48 patients | Region × atrophy is the clinical logic that EGID encodes as "distribution → type", but the paper evaluates **image-level** region and lesion accuracy only and reports **no patient-level AIG diagnosis**. The abstract's "patient-based" refers to grouping images, not to patient-level labels. |
| 10 | Chen S, Xu L, Yan L, et al. (25 authors; byline order per publisher) A novel endoscopic artificial intelligence system to assist in the diagnosis of autoimmune gastritis: a multicenter study. *Endoscopy* 2025;57(4):299–309. doi:10.1055/a-2451-3071 | Multi-site feature fusion; 18,828 images / 1,070 patients (AIG, HpAG, non-atrophic) | Patient-level AIG vs HpAG vs NAG — clinically the EGID "type" task |
| 11 | Özden Y, Omurca F. Automated diagnosis of Helicobacter pylori infection from routine endoscopic images using deep learning: a development and validation study. *BMC Gastroenterol* 2026;26:450. doi:10.1186/s12876-026-05026-7 | EfficientNet-B0, patient prediction by aggregating image predictions; 172 patients, 5-fold CV + temporal validation | Small-cohort analogue of our image-level aggregation baseline |
| 12 | Zheng W, Zhang X, Kim JJ, et al. High accuracy of convolutional neural network for evaluation of Helicobacter pylori infection based on endoscopic images: preliminary experience. *Clin Transl Gastroenterol* 2019;10(12):e00109. doi:10.14309/ctg.0000000000000109 | ResNet-50; per-image AUC 0.93 vs per-patient (8.3 ± 3.3 images) AUC 0.97 | Early evidence that multi-image patient aggregation helps |
| 13 | Seo JY, Kang J, Kim DH, et al. Clinical feasibility of deep learning-assisted classification of Helicobacter pylori infection in endoscopic imagery: a reader study. *Korean J Intern Med* 2026;41(5):862–872. doi:10.3904/kjim.2024.245 | Image- and patient-level metrics; reader study | Patient-level reporting |
| 50 | Kim S, Yoo H, Yoo S-K, Lee J, Lee H, Min YW. GutCore: an endoscopy foundation model for whole-case gastric cancer analysis. medRxiv 2026 (posted 2 July 2026). doi:10.64898/2026.07.01.26356993 — preprint, not peer reviewed | Foundation model pre-trained on 5.6 M endoscopy images; whole-case image sets of 11,035 examinations aggregated for patient-level cancer detection, invasion depth, biomarkers and survival | Whole-case (patient-level) aggregation of routine gastroscopy image sets is now also done at foundation-model scale. Cancer endpoints only, no gastritis tasks. |
| 52 | 章鑫森 (Zhang X). *Helicobacter Pylori Infection Recognition Method in Gastroscopy* (title as shown by CNKI's English interface; the original Chinese title was not captured, so look it up before citing). Master's thesis, Zhejiang University (浙江大学), supervisor 刘济全; 2022. CNKI doi:10.27461/d.cnki.gzjdx.2022.002000 — grey literature | From the CNKI record and table of contents: instance-space and hybrid-space multi-instance classification of H. pylori from gastroscopy images, plus a video system with a prospective test | Earliest patient/video-level MIL for H. pylori found (2022). Thesis, so cite only if needed. |

### 7.2 Other endoscopic deep learning for H. pylori, atrophy, distribution, AIG (S2)

| # | Reference | Relevance |
|---|---|---|
| 14 | Shichijo S, Endo Y, Aoyama K, et al. Application of convolutional neural networks for evaluating Helicobacter pylori infection status on the basis of endoscopic images. *Scand J Gastroenterol* 2019;54(2):158–163. doi:10.1080/00365521.2019.1577486 | Image-level 3-class H. pylori status (5,236 patients) |
| 15 | Li G, Togo R, Mabe K, et al. Multistage deep learning for classification of Helicobacter pylori infection status using endoscopic images. *J Gastroenterol* 2025;60(4):408–415. doi:10.1007/s00535-024-02209-5 | Subject-level 3-class status, multistage |
| 16 | Li YD, Wang HG, Chen SS, et al. Assessment of Helicobacter pylori infection by deep learning based on endoscopic videos in real time. *Dig Liver Dis* 2023;55(5):649–654. doi:10.1016/j.dld.2023.02.010 | Video-level H. pylori |
| 17 | Li F, Li Z, Wu W, et al. AI-assisted endoscopic detection of Helicobacter pylori infection in atrophic gastritis: a multicenter diagnostic study. *Surg Endosc* 2026 (online 19 Aug 2026; volume/pages still unassigned in Crossref and PubMed on 2026-10-19, so cite as online-first). doi:10.1007/s00464-026-13064-1 | H. pylori detection degrades in atrophic mucosa — interaction of our two independent tasks |
| 18 | Tao X, Zhu Y, Dong Z, et al. An artificial intelligence system for chronic atrophic gastritis diagnosis and risk stratification under white light endoscopy. *Dig Liver Dis* 2024;56(8):1319–1326. doi:10.1016/j.dld.2024.01.177 | Patient-level atrophy **extent** stratification (Kimura-Takemoto: none/mild/moderate/severe) |
| 19 | Zhao Q, Jia Q, Chi T. U-Net deep learning model for endoscopic diagnosis of chronic atrophic gastritis and operative link for gastritis assessment staging: a prospective nested case-control study. *Ther Adv Gastroenterol* 2023;16:17562848231208669. doi:10.1177/17562848231208669 | OLGA staging from video |
| 20 | Kodaka Y, Futagami S, Watanabe Y, et al. Determination of gastric atrophy with artificial intelligence compared to the assessments of the modified Kyoto and OLGA classifications. *JGH Open* 2022;6(10):704–710. doi:10.1002/jgh3.12810 | AI atrophy score vs Kimura-Takemoto/Kyoto/OLGA |
| 21 | Song Y, Zhang X, Shi L, et al. Artificial intelligence for single-image diagnosis of autoimmune gastritis using gastric corpus atrophy images: a multicenter retrospective diagnostic study. *J Gastroenterol Hepatol* 2026;41(4):1292–1302. doi:10.1111/jgh.70296 | AIG vs HpAG from one corpus image; 2,137 patients |
| 22 | Cao Y, Jin X, Zhao Y, et al. A multimodal artificial intelligence method for accurate diagnosis of autoimmune gastritis. *Dig Liver Dis* 2026;58(9):1156–1164. doi:10.1016/j.dld.2026.07.005 | Images + pathology report + laboratory tests |
| 23 | Yang F, Xue N, Jiang N, et al. Deep learning model based on gastroscopic images can differentiate between Helicobacter pylori gastritis and autoimmune gastritis. *Eur J Med Res* 2026;31:989. doi:10.1186/s40001-026-04536-0 | Image-level AIG vs HpG (525 images) |
| 24 | Lin N, Yu T, Zheng W, et al. Simultaneous recognition of atrophic gastritis and intestinal metaplasia on white light endoscopic images based on convolutional neural networks: a multicenter study. *Clin Transl Gastroenterol* 2021;12(8):e00385. doi:10.14309/ctg.0000000000000385 | Joint AG + IM image-level recognition |
| 25 | Liu A, Zhang X, Zhong J, et al. A deep learning approach for gastroscopic manifestation recognition based on Kyoto Gastritis Score. *Ann Med* 2024;56(1):2418963. doi:10.1080/07853890.2024.2418963 | Multi-label Kyoto score, independent heads per finding |
| 26 | Xiong Z, Chen S, Zhang Y, et al. Adaptify: a refined test-time adaptation scheme for frame classification consistency in atrophic gastritis videos. *2024 IEEE Int Symp Biomed Imaging (ISBI)*; 2024:1–5. doi:10.1109/ISBI56570.2024.10635341 | Frame-level consistency in atrophy videos (test-time adaptation) |
| 27 | Yang J, Ou Y, Chen Z, et al. A benchmark dataset of endoscopic images and novel deep learning method to detect intestinal metaplasia and gastritis atrophy. *IEEE J Biomed Health Inform* 2023;27(1):7–16. doi:10.1109/JBHI.2022.3217944 | Public image-level IM/atrophy dataset (21,420 images) |
| 28 | Jiang Y, Yan H, Cui J, et al. Artificial intelligence in endoscopy for predicting Helicobacter pylori infection: a systematic review and meta-analysis. *Helicobacter* 2025;30(2):e70026. doi:10.1111/hel.70026 | Meta-analysis, 16 studies |
| 29 | Parkash O, Lal A, Subash T, et al. Use of artificial intelligence for the detection of Helicobacter pylori infection from upper gastrointestinal endoscopy images: an updated systematic review and meta-analysis. *Ann Gastroenterol* 2024;37(6):665–673. doi:10.20524/aog.2024.0913 | Meta-analysis |
| 30 | Shi Y, Wei N, Wang K, et al. Diagnostic value of artificial intelligence-assisted endoscopy for chronic atrophic gastritis: a systematic review and meta-analysis. *Front Med* 2023;10:1134980. doi:10.3389/fmed.2023.1134980 | Meta-analysis, 8 studies |
| 31 | Xia K, Hu Y, Cai S, et al. GastritisMIL: an interpretable deep learning model for the comprehensive histological assessment of chronic gastritis. *Patterns* 2025;6(8):101286. doi:10.1016/j.patter.2025.101286 | Multi-task MIL for gastritis — but on **histology** WSIs (contrast only) |
| 51 | Jeong WC, Kim HH, Hwang Y, Hwang G, Kim K, Ko YS. SydneyMTL: interpretable multi-task learning for complete Sydney System assessment in gastric biopsies. medRxiv 2026 (posted 18 Feb 2026). doi:10.64898/2026.02.17.26346304 — preprint | Multi-task MIL with **task-specific attention pooling** for all five Updated Sydney System attributes (incl. H. pylori, atrophy) on 50,765 biopsy WSIs. Histology, contrast only, but it is direct precedent for our task-specific attention branches. |

### 7.3 EGID (S3)

| # | Reference | Relevance |
|---|---|---|
| 32 | Li Z, Fang Y, Song Y, Bai H, Li T, Wang Y, Zhang Q, Li X. EGID: a comprehensive multi-label endoscopic image dataset for gastritis classification. *Sci Data* 2026;13:1396 (published 22 June 2026). doi:10.1038/s41597-026-07666-6 | Dataset and reference benchmark |
| 33 | Tran T, Dang L. woma: a real-time foundation model and its fine-tuned models for endoscopy. arXiv:2609.15130 (2026) | Only other EGID user found; uses EGID images as pre-training/fine-tuning data, no gastritis results |

### 7.4 Structured / hierarchical label consistency in medical classification (S4)

| # | Reference | Relevance |
|---|---|---|
| 34 | Chen H, Miao S, Xu D, Hager GD, Harrison AP. Deep hiearchical multi-label classification applied to chest X-ray abnormality taxonomies. *Med Image Anal* 2020;66:101811. doi:10.1016/j.media.2020.101811 (sic, "hiearchical" in the published title) | Conditional-probability training respecting a clinical taxonomy |
| 35 | Ju L, Yu Z, Wang L, et al. Hierarchical knowledge guided learning for real-world retinal disease recognition. *IEEE Trans Med Imaging* 2024;43(1):335–350. doi:10.1109/TMI.2023.3302473 | Hierarchy-guided multi-label fundus |
| 36 | Yang Z, Li Y, Liu Y. Hierarchy-aware and knowledge-guided learning for multi-label classification of retinal diseases from fundus images. *IEEE Trans Med Imaging* 2026;45(6):3262–3275. doi:10.1109/TMI.2026.3672477 | Explicit hierarchy-level consistency |
| 37 | Wang T, Hou Q, Cao P, et al. C²Net: a co-occurrence and consistency-aware framework for structured multi-label fundus diagnosis. *Med Biol Eng Comput* 2026;64(6):2209–2223. doi:10.1007/s11517-026-03574-4 | Penalises logically conflicting outputs (e.g. "normal" + disease) |
| 38 | Xu Y, Wen G, Yang P, et al. Task-coupling elastic learning for physical sign-based medical image classification. *IEEE J Biomed Health Inform* 2022;26(2):626–637. doi:10.1109/JBHI.2021.3106837 | Coupled location→nature tasks |
| 39 | Lee C, Nguyen Quoc K, Song J, Chong Y, Yim K, Kwak JT. TaxoMIL: taxonomy-constrained learning for hierarchical whole slide image analysis. arXiv:2606.31100 (2026) | **MIL + taxonomy constraints** (pathology) |
| 40 | Strong J, Saha P, Sun E, Higham H, Noble A. Coherent hierarchical multi-label learning to defer for medical imaging. arXiv:2605.02734 (2026) | Measures/prevents incoherent hierarchical outputs |
| 41 | Zhang P, Wu Y, Hu X, et al. Hierarchical multi-instance learning for targeted triage and classification of common left-to-right shunt lesions from echocardiography videos. *BMC Med Imaging* 2026;26:473. doi:10.1186/s12880-026-02544-4 | Exam-level attention MIL over a variable number of clips + hierarchical outputs |

### 7.5 Instance dropout / pseudo-bags / missing views in MIL (S5)

| # | Reference | Relevance |
|---|---|---|
| 42 | Zhang H, Meng Y, Zhao Y, et al. DTFD-MIL: double-tier feature distillation multiple instance learning for histopathology whole slide image classification. *Proc IEEE/CVF CVPR* 2022:18780–18790. doi:10.1109/CVPR52688.2022.01824 | Random pseudo-bags (instance subsets inherit the bag label) |
| 43 | Tang W, Huang S, Zhang X, et al. Multiple instance learning framework with masked hard instance mining for whole slide image classification. *Proc IEEE/CVF ICCV* 2023:4055–4064. doi:10.1109/ICCV51070.2023.00377 | Masks high-attention instances; **teacher–student consistency** between full and masked bags |
| 44 | Zhang Y, Li H, Sun Y, Zheng S, Zhu C, Yang L. Attention-challenging multiple instance learning for whole slide image classification. In: *Computer Vision – ECCV 2024*, LNCS. Springer; 2025:125–143. doi:10.1007/978-3-031-73668-1_8 | Random masking of top-attended instances against attention over-concentration |
| 45 | Liu P, Ji L, Zhang X, Ye F. Pseudo-bag mixup augmentation for multiple instance learning-based whole slide image classification. *IEEE Trans Med Imaging* 2024;43(5):1841–1852. doi:10.1109/TMI.2024.3351213 | Pseudo-bag subsampling and mixing |
| 46 | Dasdelen MF, Ozlugedik F, Litinetskaya A, Navab N, Marr C, Sadafi A. Re-mixing embeddings for patient augmentation in data scarce multiple instance learning. arXiv:2606.25770 (2026) | Patient-level augmentation for small MIL cohorts |
| 47 | Zhao C, Tang H, Zhao H, Li X. Diffmv: a unified diffusion framework for healthcare predictions with random missing views and view laziness. arXiv:2505.11802 (2025) | Training for random missing views (EHR) |
| 48 | Neverova N, Wolf C, Taylor G, Nebout F. ModDrop: adaptive multi-modal gesture recognition. *IEEE Trans Pattern Anal Mach Intell* 2016;38(8):1692–1706. doi:10.1109/TPAMI.2015.2461544 | Seed; input-dropping precedent (pre-2018, not searched) |
| 49 | Shiku K, Nishimura K, Suehiro D, Tanaka K, Bise R. Ordinal multiple-instance learning for ulcerative colitis severity estimation with selective aggregated transformer. *Proc IEEE/CVF WACV* 2025:4290–4299. doi:10.1109/WACV61041.2025.00421 | Endoscopy MIL with structured (ordinal) outputs, colon |


### 7.6 Verification of the references already in the repository

| Reference as previously listed | Verified record | Change needed |
|---|---|---|
| Li Z, Fang Y, Song Y, et al. *Sci Data* 13, 1396 (2026), doi:10.1038/s41597-026-07666-6 | Same; 8 authors; online 22 June 2026; vol 13, issue 1, art. 1396 | none |
| EGID benchmark numbers (backbones, DenseNet121 per task, distribution CI 58.75–88.24) | Match Li et al. Tables 4 and 5 (checked on nature.com) | none |
| Ilse M, Tomczak J, Welling M. *ICML* 2018 | *Proc 35th ICML*, PMLR 80:2127–2136 | add pages |
| Campanella G, et al. *Nat Med* 2019 | 25(8):1301–1309, doi:10.1038/s41591-019-0508-1 | add volume/DOI |
| Lu MY, et al. (CLAM) *Nat Biomed Eng* 2021 | 5(6):555–570, doi:10.1038/s41551-020-00682-w | add volume/DOI |
| Shao Z, et al. TransMIL. *NeurIPS* 2021 | *Adv Neural Inf Process Syst* 34:2136–2147 | add volume/pages |
| Giunchiglia E, Lukasiewicz T. *NeurIPS* 2020 | *Adv Neural Inf Process Syst* 33:9662–9673 | add volume/pages |
| Xu J, et al. Semantic loss. *ICML* 2018 | Xu J, Zhang Z, Friedman T, Liang Y, Van den Broeck G. PMLR 80:5502–5511 | add pages |
| Neverova N, et al. ModDrop. *IEEE TPAMI* 2016 | 38(8):1692–1706, doi:10.1109/TPAMI.2015.2461544 | add volume/DOI |
| Hinton G, Vinyals O, Dean J. 2015 | arXiv:1503.02531 | add arXiv ID |
| Sohn K, et al. FixMatch. *NeurIPS* 2020 | *Adv Neural Inf Process Syst* 33:596–608 | add volume/pages |
| Guo C, et al. *ICML* 2017 | Guo C, Pleiss G, Sun Y, Weinberger KQ. PMLR 70:1321–1330 | add pages |
| Lin T-Y, et al. Focal loss. *ICCV* 2017 | *Proc IEEE ICCV* 2017:2999–3007, doi:10.1109/ICCV.2017.324 | add pages/DOI |
| Shichijo S, et al. *EBioMedicine* 2017 | 25:106–111, doi:10.1016/j.ebiom.2017.10.014 | add volume/DOI |
| Itoh T, et al. *Endosc Int Open* 2018 | 6(2):E139–E144, doi:10.1055/s-0043-120830 | add volume/DOI |

No existing citation was wrong; all were incomplete (missing volume/pages/DOI). One database
discrepancy was found and resolved: Crossref lists the authors of Chen S et al. (*Endoscopy* 2025,
#10) as "Chen S, Xu L, Li T, Chen Y, Yan L, …" whereas PubMed lists "Chen S, Xu L, Yan L, …".
**Resolved 2026-10-19:** the publisher's own record (Thieme Connect `citation_author` metadata)
gives the byline "Shurong Chen; Louzhe Xu; Lingling Yan; Jie Zhang; Xuefeng Zhou; …", the same as
PubMed. Crossref's record moves the four authors tagged `sequence="first"` (equal-contribution
first authors: Chen S, Xu L, Li T, Chen Y) to the front. Cite as **Chen S, Xu L, Yan L, et al.**

### 7.7 Full-text verification of the closest prior work (#1–#10)

Open-access status was checked via OpenAlex `open_access` and Europe PMC, and a preprint search
was run on arXiv and Europe PMC (2026-10-19). Full texts came from Europe PMC `fullTextXML`.

| # | Full text? | What the full text shows (relevant to our claims) |
|---|---|---|
| 1 PhenoMIL | No: closed (Elsevier), no preprint found | Abstract only |
| 2 HOPE AI | **Yes** (PMC12505616 + Additional files 1–2) | Top-k mean pooling (k = 7) + LSTM on PVT features, image + patient losses; single task (H. pylori +/−); inclusion required ≥ 8 standard gastric images; 34.7 ± 15.7 images per training case. In-house comparison (internal validation, Suppl. Table 1): ABMIL AUC 87.62, DSMIL 90.93, TransMIL 87.83 vs their top-k baseline 92.12. **No experiment with fewer or missing images**; no multi-label structure. |
| 3 Ting et al. | No: closed (IEEE); the IEEE page shows only the abstract and the start of the introduction | Abstract + first page only |
| 4 DwinTMIL | No: closed (IEEE), no preprint | Abstract only |
| 5 EndoStrat | **Yes** (PMC13561896) | ResNet-50 features + **gated attention MIL (Ilse et al.)**, the same pooling as our A1; two-stage, cost-sensitive. Image-level comparator = per-image classifier + **max**-probability pooling (ours is mean). Internal-validation AUC: EndoStrat 0.770, 4-class MIL 0.659, image-level 0.602. Inclusion required ≥ 20 images (median 25, IQR 22–28). Training cohort 3,555 patients. **No fewer-image experiment**; one label per patient (highest-grade pathology), no inter-task structure. |
| 6 Dai J et al. | No: closed (Springer), no preprint | Abstract only (single- vs multi-image comparison is stated there) |
| 7 Dai D et al. | No: closed (Wiley), no preprint | Abstract only (single-site vs multisite comparison stated) |
| 8 GSCNet | **Yes** (PMC11186652) | Input is a **fixed triplet of one antrum, one body and one cardia image** per patient; three section-correlation losses encode the clinical definition of CGI (antrum vs corpus); binary CGI output; single random train/test split, no validation set (fixed epochs). The patient counts in Table I cannot be reconciled with the reported test metrics, so they are not quoted. **No handling of missing sections**, no multi-task output. |
| 9 Cao et al. | **Yes** (PMC12112349) | See row #9: image-level region + atrophy labels and image-level evaluation only; **no patient-level AIG diagnosis reported**. The earlier summary ("patient-based multi-task AIG diagnosis") was corrected. |
| 10 Chen S et al. | No: closed (Thieme), no preprint | Abstract only |

Status of each negative or comparative claim used in `literature_comparison.md` and
`research_gap.md`. **FT** = verified in the full text; **abstract** = rests on the abstract only.

| Claim | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Patient-level / bag-level model on gastroscopy images | abstract | FT | abstract | abstract | FT | abstract | abstract | FT (fixed 3-image input) | **FT: no**, image-level | abstract |
| Does not train or test with fewer / missing images (no degradation curve) | abstract | FT | abstract | abstract | FT | abstract: compares 1 image vs all only | abstract: compares 1 site vs all only | FT | FT | abstract |
| Does not model coupled / deterministic label structure or report a coherence rate | abstract (has phenotype → diagnosis prior) | FT | abstract | abstract | FT | abstract | abstract | FT (uses a section prior, single output) | FT (two independent image tasks) | abstract |
| Needs per-image anatomical labels or a fixed per-section input | — | — | — | — | — | abstract (multi-region) | abstract (multisite) | FT | FT | abstract (multi-site) |

## 8. Limitations of this search and open items

Limitations that remain after the second round:

1. **Single screener.** Most records were screened on title, with keyword-assisted abstract
   screening and 54 abstracts read in full. Full texts were read for 4 of the 10 closest papers;
   6 are paywalled with no preprint (#1, #3, #4, #6, #7, #10). Negative claims about those six
   rest on abstracts (§7.7).
2. IEEE Xplore was searched without a subscription: abstracts are snippets, and only the first 100
   records per query are exported (all queries but I5 had ≤100 hits; I5's 89 were all exported).
3. **Google Scholar** coverage is one query with the first 10 of ≈70 results; the second query hit
   a CAPTCHA and was not run (§3a).
4. **CNKI**: five subject queries; only the first page of C1/C5 was screened, titles only, and a
   slider verification on record pages stopped further reading (§3a). Wanfang and VIP (Chinese)
   and other non-English databases were not searched.
5. Preprints: arXiv and medRxiv (via Europe PMC) were searched; Research Square and SSRN were not,
   except where OpenAlex surfaced a record (one Research Square preprint, excluded).
6. M3JLBN (Lai et al. 2022) could not be assessed (§6).

**Open items for the user** (each needs credentials, a second person, or a later date):

- [ ] Run O1–O5 (as `TITLE-ABS-KEY`) in **Scopus**, and the PubMed strings in **Embase** and
      **Web of Science**, using institutional login; add counts to §3 and screen new records.
- [ ] Get the full texts of #1, #3, #4, #6, #7, #10 (and M3JLBN) through the library and update
      the "abstract" cells of §7.7.
- [ ] **Second screener**: independently screen at least the 54 abstracts plus the §7 inclusions,
      and report agreement.
- [ ] Optional: finish Google Scholar G2 (and further pages of G1) and CNKI C1/C5 page 2
      interactively in a normal browser.
- [ ] **Pre-submission re-run** of P7/A3/I4/O4/M3, the EGID citation look-ups (§4), and P1/A1/I1/O1
      (EGID was published 2026-06-22; MIL-on-gastroscopy papers are appearing monthly).
- [ ] Look up the original Chinese title of #52 before citing it.
- [ ] Commit the `research/` changes.
