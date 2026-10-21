# CoRe-MIL on EGID

Constraint-aware, missing-view-robust multiple-instance learning for patient-level multi-label
gastritis classification on the Endoscopic Gastritis Image Dataset (EGID; Li et al., *Sci Data*
2026). Research code: audit → splits → features → baselines/ablations → statistics → figures.

The proposed architecture is a research hypothesis. Results are only what `outputs/` contains;
see `research/` for the protocol, methodology, research gap and limitations.

## Data

The EGID images are not included. Download them from figshare (CC BY 4.0;
[part 1](https://doi.org/10.6084/m9.figshare.30996127), [part 2](https://doi.org/10.6084/m9.figshare.30997345);
dataset paper: Li et al., *Sci Data* 2026, https://doi.org/10.1038/s41597-026-07666-6) and place the
extracted folders and `metadata.xlsx` in `data/`. The code never modifies `data/`.

## Licence

Code: MIT (see `LICENSE`). Derived outputs follow the dataset's CC BY 4.0 licence.

## Layout

```text
data/                     EGID as released (read-only; never modified)
configs/mac_m1.yaml       hardware profile, paths, run profiles (smoke/dev/full/encoders), fine-tuning
configs/experiments.yaml  protocol, encoders, hyper-parameters, variants (baselines + ablations), groups
src/data/                 metadata parsing, audit, splits, image cache, bags/view sampling
src/models/               encoders (timm), image-level baseline, MIL + task attention, structured head
src/training/             losses, cached-feature trainer, feature extraction, end-to-end fine-tuning
src/evaluation/           metrics, temperature scaling, missing-view subsets, statistics, aggregation
src/visualization/        Figure 1 schematic and Figures 2-8
scripts/                  audit_data, prepare_data, extract_features, tune, finetune, evaluate, generate_paper_figures
run_experiments.py        grid runner (variants x encoders x splits x seeds), resumable
tests/                    pytest suite (CPU)
outputs/                  data_audit/, splits/, tuning/, runs/, predictions/, attention/, history/,
                          metrics/, tables/, figures/, experiments.jsonl
research/                 experiment_protocol, methodology, research_gap, literature_comparison, limitations
```

## Setup (Apple M1, macOS)

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Pretrained weights are downloaded by `timm` from the Hugging Face hub on first use (~0.5 GB for all six encoders).

## Pipeline

```bash
.venv/bin/python scripts/audit_data.py
```

Verifies folder/metadata agreement, hashes every file (the four task folders hold byte-identical
copies), detects corrupt files and duplicates, and writes `outputs/data_audit/` (report:
`dataset_audit.md`). ~40 s.

```bash
.venv/bin/python scripts/prepare_data.py --sizes 224 256
```

Writes patient-level splits (`outputs/splits/`: `primary` 161/34/34 matching the EGID Table 3
composition, `cv0..cv4` inside the 195-patient development cohort) and uint8 memmap image caches
(`cache/images_224.npy`, 0.9 GB; 256 for fine-tuning). ~1 min. Saved splits are never overwritten
with different content.

```bash
.venv/bin/python scripts/extract_features.py
```

Frozen embeddings for EfficientNet-B0, DenseNet121, ResNet50, ConvNeXt-T, Swin-T, DINOv2-S on MPS
(`cache/features/`). 25–60 s per encoder.

```bash
.venv/bin/python -m pytest -q tests
```

```bash
.venv/bin/python run_experiments.py --profile smoke --group all
```

Smoke test: every variant, 3 epochs, 24 patients, written to `outputs/smoke/` (numbers meaningless).

```bash
.venv/bin/python scripts/tune.py --select
```

Hyper-parameter selection on inner-validation NLL (run the grid first: `scripts/tune.py --variants IMG`,
`--variants A0`, `--variants A1`; add `--folds cv0` etc. to split it over processes). The selection is
copied into `configs/experiments.yaml` under `tuned:`.

```bash
.venv/bin/python run_experiments.py --profile full --group all --splits cv0 cv1 cv2 cv3 cv4
```

Main CV grid (13 variants × 5 folds × 5 seeds). Resumable: completed runs are skipped; a run whose
configuration changed raises instead of being silently reused (`--force` archives the old record).
To use several cores, start one process per seed (`--seeds 0`, `--seeds 1`, …).

```bash
.venv/bin/python run_experiments.py --profile full --variants IMG --encoders resnet50 densenet121 --splits cv0 cv1 cv2 cv3 cv4
```

```bash
.venv/bin/python run_experiments.py --profile encoders --group encoders
```

```bash
.venv/bin/python scripts/evaluate.py --scheme cv
```

```bash
.venv/bin/python scripts/generate_paper_figures.py --scheme cv
```

Untouched test cohort, **once**, after the CV analysis (requires the explicit flag):

```bash
.venv/bin/python run_experiments.py --profile full --group all --splits primary --allow-test
```

```bash
.venv/bin/python scripts/evaluate.py --scheme test
```

Optional end-to-end fine-tuning on MPS (FT_IMG reproduces the EGID-paper training protocol as far as
it is described; FT_A6 is end-to-end CoRe-MIL):

```bash
.venv/bin/python scripts/finetune.py --variant FT_IMG --encoder densenet121 --splits cv0 cv1 cv2 cv3 cv4
```

```bash
.venv/bin/python scripts/finetune.py --variant FT_A6 --encoder efficientnet_b0 --splits cv0 cv1 cv2 cv3 cv4
```

Grad-CAM for a fine-tuned CoRe-MIL fold (supplementary figure):

```bash
.venv/bin/python scripts/gradcam.py --split cv0
```

## Final model (your trained CoRe-MIL)

`models/coremil_final.pt` is the full CoRe-MIL (A6) trained on all 229 patients; see
`models/MODEL_CARD.md` for how it was trained and its expected (cross-validated) performance.
Predict one patient from a folder of their images:

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --json result.json
```

The strongest CNN (image-level DenseNet121, CV mean accuracy 79.5%) is exported the same way as
`models/cnn_densenet121_final.pt`:

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --model models/cnn_densenet121_final.pt
```

The end-to-end fine-tuned CoRe-MIL (CV distribution macro-F1 0.617, below the DenseNet121 model) is
`models/coremil_ft_final.pt`:

```bash
.venv/bin/python scripts/predict.py path/to/one_patient_folder --model models/coremil_ft_final.pt
```

Rebuild CoRe-MIL (refuses to overwrite an existing file):

```bash
.venv/bin/python scripts/train_final.py
```

## Variants

| ID | Pooling | Head | View dropout | Consistency | Paper role |
|---|---|---|---|---|---|
| IMG | image-level, mean of probabilities | independent | – | – | B1/B2/B3 with ResNet50 / EfficientNet-B0 / DenseNet121 |
| A0 | mean | independent | – | – | mean-pooling MIL |
| B4 | one gated attention | independent | – | – | attention MIL |
| A1 | task-aware gated attention | independent | – | – | B5 |
| A2 | task-aware | structured | – | – | B6 |
| A3 | task-aware | independent | ✓ | – | ablation |
| A4 | task-aware | independent | ✓ | ✓ | ablation |
| A5 | task-aware | structured | ✓ | – | B7 |
| A6 | task-aware | structured | ✓ | ✓ | B8, full CoRe-MIL |
| X1–X4 | as A1/A6 | | | | soft clinical loss; weighted CE; focal; balanced sampling |

The structured head is used because the audit found atrophy presence and gastritis type to be exact
functions of atrophy distribution; the mapping is re-derived on each split's training patients and
the code falls back to independent heads if it is not deterministic there.

## Mac M1 notes

- Feature extraction and fine-tuning use MPS (`device: auto`). Cached-feature MIL training runs on
  **CPU** (`mil_device: cpu`): for these tiny models CPU was ~20× faster than MPS in a benchmark
  (A6, 20 epochs: 2.8 s CPU vs 55 s MPS) because MPS per-kernel launch overhead dominates.
- No silent CPU fallback: `PYTORCH_ENABLE_MPS_FALLBACK` is not set; if you set it, a warning is logged.
- `num_workers: 0` (macOS spawn); images come from a memmap, so RAM stays < 3 GB for frozen runs.
- MPS out-of-memory errors are re-raised with guidance: lower `feature_extraction.batch_size`,
  `finetune.batch_images` or `finetune.bag_size_train`; increase `grad_accum`.
- Peak disk use: caches ~2 GB, features ~0.1 GB, predictions/records < 1 GB.

## Reproducibility

Each run writes `outputs/runs/<run_id>.json` with seed, split, resolved configuration and its hash,
encoder, hyper-parameters, loss coefficients, validation and held-out metrics at every image
fraction, temperatures, the label mapping used, checkpoint path, runtime, library versions and a
content hash of the code (`git` commit when the project is its own repository). Per-epoch histories
are in `outputs/history/`, patient-level predictions in `outputs/predictions/`, attention weights in
`outputs/attention/`, and one summary line per run is appended to `outputs/experiments.jsonl`.

## Status (2026-10-19)

| Item | Status |
|---|---|
| Audit, label-relationship audit, splits, caches, features (6 encoders) | done |
| Tests (`pytest`, 37 tests) and smoke run on real data | pass |
| Hyper-parameter grid (inner validation only) | done, `outputs/tuning/` |
| Pre-specified CV grid: 13 variants + B1/B3, 5 folds × 5 seeds | done (375 runs) |
| Encoder sweep: IMG/A0/A1/A6 × 6 encoders × 5 folds × 3 seeds | done |
| Post-hoc mean-pooling variants M2/M3/M5/M6 (exploratory) | done (CV + test) |
| Primary test cohort, evaluated once | done (95 runs) |
| Tables 1–8 + S1, Figures 1–8 (CV and test) | done, `outputs/tables`, `outputs/figures` |
| End-to-end fine-tuning FT_IMG (DenseNet121, 5 folds, 1 seed) | done: CV distribution macro-F1 0.614 [0.536, 0.685] |
| End-to-end fine-tuning FT_A6 (EffNet-B0, 5 folds, 3 seeds) | done: CV distribution macro-F1 0.617 [0.558, 0.671] |
| Addendum A: improved fine-tuning FT2_IMG (DenseNet121/EffNet-B0/ResNet50, 2 seeds) | done: 0.703 / 0.730 / 0.648, none differs from frozen |
| Addendum B: top-k, attention masking, transformer MIL (N1–N6) | done, conclusion unchanged |
| Final fine-tuned CoRe-MIL on all 229 patients | done, `models/coremil_ft_final.pt` |
| Addendum C: image-level loss / augmented features for CoRe-MIL (C1–C3, CV, 5 seeds) | done: primary C3 − A6 +0.027 [−0.009, 0.063], not significant; C1 alone 0.637 |
| Grad-CAM for FT_A6 (`scripts/gradcam.py`) | done, `outputs/figures/cv/figS_gradcam_ft_a6.*` |

Headline (CV, n=195): the pre-specified CoRe-MIL (A6) did **not** outperform image-level aggregation
or mean-pooling MIL (distribution macro-F1 0.580 vs 0.722 / 0.677). See `research/technical_report.md`.
