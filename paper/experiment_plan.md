# Experiment Plan

## Model Naming Convention

| Model Name | Description |
|---|---|
| `vision` | Morphology-only (ViT fine-tuned end-to-end) |
| `CONCH` | Morphology-only foundation model baseline (frozen) |
| `Geneformer` | Expression-only foundation model baseline (frozen) |
| `expr-tile` | Expression-only, tile-level transcript aggregation |
| `expr-token` | Expression-only, token-level transcript aggregation |
| `early-fusion-add` | Early fusion, element-wise addition |
| `early-fusion-concat` | Early fusion, concatenation |
| `late-fusion-tile-add` | Late fusion, tile-level transcripts, addition |
| `late-fusion-tile-concat` | Late fusion, tile-level transcripts, concatenation |
| `late-fusion-token-add` | Late fusion, token-level transcripts, addition |
| `late-fusion-token-concat` | Late fusion, token-level transcripts, concatenation |

## Model Complexities

| Complexity | Vision Backbone | Expr Encoder |
|---|---|---|
| low | ViT-S | 1-layer MLP |
| high | ViT-B | 3-layer MLP |

## Datasets

| Dataset | Description |
|---|---|
| BEAT | Internal cancer cohort (159 patients, Xenium) |
| HEST1k | Public multi-tissue ST dataset |
| HESCAPE | Benchmark subset of HEST1k samples |

## Gene Panels

| Panel | Description |
|---|---|
| HVG-50 | 50 most highly variable genes |
| HESCAPE panel | Panel genes as defined by the HESCAPE benchmark |

## Evaluation Metrics

- MSE
- Pearson correlation coefficient (per-gene / per-cell-type, then averaged)
- Spearman correlation coefficient (per-gene / per-cell-type, then averaged)
- Sample-level Pearson (average tiles per sample, then average across samples — equal-weighted)

## Experiments

### A. Gene Expression Prediction

| Experiment | Dataset(s) | Complexity | Panel | Status | Responsible |
|---|---|---|---|---|---|
| A1. All models, BEAT, HVG-50, low capacity | BEAT | low | HVG-50 | completed | AM |
| A2. All models, BEAT, HVG-50, high capacity | BEAT | high | HVG-50 | completed | AM |
| A3. All models, BEAT, HESCAPE panel, low capacity | BEAT | low | HESCAPE | not-started | AM |
| A4. All models, BEAT, HESCAPE panel, high capacity | BEAT | high | HESCAPE | not-started | AM |
| A5. All models, HEST1k, HVG-50, low capacity | HEST1k | low | HVG-50 | not-started | AM |
| A6. All models, HEST1k, HVG-50, high capacity | HEST1k | high | HVG-50 | not-started | AM |
| A7. All models, HEST1k, HESCAPE panel, low capacity | HEST1k | low | HESCAPE | not-started | AM |
| A8. All models, HEST1k, HESCAPE panel, high capacity | HEST1k | high | HESCAPE | not-started | AM |
| A9. All models, HESCAPE, HVG-50, low capacity | HESCAPE | low | HVG-50 | not-started | AM |
| A10. All models, HESCAPE, HVG-50, high capacity | HESCAPE | high | HVG-50 | not-started | AM |
| A11. All models, HESCAPE, HESCAPE panel, low/high capacity | HESCAPE | low+high | HESCAPE | not-started | AM |

### B. Cell Type Composition Prediction

| Experiment | Dataset(s) | Complexity | Panel | Status | Responsible |
|---|---|---|---|---|---|
| B1. All models, BEAT, HVG-50, low capacity | BEAT | low | HVG-50 | completed | AM |
| B2. All models, BEAT, HVG-50, high capacity | BEAT | high | HVG-50 | completed | AM |
| B3. All models, BEAT, HESCAPE panel | BEAT | low+high | HESCAPE | not-started | AM |
| B4. All models, HEST1k, HVG-50 | HEST1k | low+high | HVG-50 | not-started | AM |
| B5. All models, HESCAPE, HESCAPE panel | HESCAPE | low+high | HESCAPE | not-started | AM |

### C. Per-Sample vs. Per-Gene/Per-Cell-Type Evaluation

| Experiment | Description | Status | Responsible |
|---|---|---|---|
| C1. Per-gene tile-level Pearson on BEAT | Already computed; export for preprint | completed | AM |
| C2. Sample-level (equal-weighted) Pearson on BEAT | Average tiles per sample, then avg across samples | ongoing | AM |
| C3. Sample-level evaluation on HEST1k / HESCAPE | As above, extended to public datasets | not-started | AM |

### D. Clinical Downstream Tasks (MIL)

| Experiment | Description | Status | Responsible |
|---|---|---|---|
| D1. Histology classification from embeddings | Train MIL classifier on top of model embeddings | not-started | AM |
| D2. Survival prediction from embeddings | Train MIL survival model (e.g., ABMIL) on embeddings | not-started | AM |
| D3. Comparison across all model variants | Compare MIL performance across vision, expr, fusion variants | not-started | AM |

### E. Spatial Visualization

| Experiment | Description | Status | Responsible |
|---|---|---|---|
| E1. Gene expression spatial maps (BEAT) | True vs. predicted per gene on representative WSI | completed | AM |
| E2. Cell type abundance spatial maps (BEAT) | True vs. predicted per cell type on representative WSI | completed | AM |
| E3. Spatial maps on HEST1k / HESCAPE samples | Generalization of spatial predictions | not-started | AM |

### F. CLIP Contrastive Pretraining

| Experiment | Description | Status | Responsible |
|---|---|---|---|
| F1. CLIP-tile vs. CLIP-token on BEAT | Cross-modal retrieval recall@k | completed | AM |
| F2. Transfer: CLIP-pretrained ViT-S → fine-tune | Gene prediction with CLIP-pretrained backbone | ongoing | AM |

### G. Panel Sensitivity Analysis

| Experiment | Description | Status | Responsible |
|---|---|---|---|
| G1. Core vs. add-on vs. full panel (cell type task) | Completed for BEAT | completed | AM |
| G2. Panel sensitivity on HEST1k | Not-started | not-started | AM |

---

## Notes

- "HESCAPE" defines a curated subset of HEST1k samples with a standardized panel; it serves both as a dataset and as a panel definition.
- Per-sample equal-weighted scores are important because samples differ widely in tile count; tile-weighted averages may be dominated by large WSIs.
- MIL downstream tasks (D) require embeddings from all model variants; run E experiments after A/B are complete.
- Model naming should be consistent throughout all figures, tables, and text: use the names in the table above verbatim.
