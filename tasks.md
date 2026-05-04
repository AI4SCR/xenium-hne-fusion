# Tasks

## Status legend

- 🟢 `completed`
- 🟡 `running`
- 🟠 `in preparation`
- 🔴 `open`

## Current status

### 🟢 Completed

- `BEAT`: cell type run with the default panel
- `BEAT`: expression run with the default panel
- `HEST1k` with HESCAPE splits and panel:
  - `lung-healthy`
  - `human-immuno-oncology`
  - `bowel`
  - `breast`
  - `human-multi-tissue`

### 🟡 Running

- `BEAT`: runs with `50` HVG genes
- `BEAT`: runs with `100` HVG genes

### 🟠 In preparation

- MIL for regression
- MIL for survival
- MIL for classification (`histology`)

### 🔴 Open

- Re-run `cell_types` predictions with **unfrozen** `morph_encoder` — model checkpoints needed for MIL experiments (previous runs on Ray cluster did not save checkpoints)
- Re-run unimodal and late-fusion experiments for **frozen** foundation model encoders: `UNI`, `CONCH`, `BIOPTIMUS`, `GENEFORMER` — code exists, needs smoke-test; path forward is to add configs that set `morph_encoder` to each foundation model. Known hiccup: `PHIKON` checkpoint needs to be downloaded. Likely other small hiccups (missing checkpoints, env vars, etc.) but no major code changes should be necessary.
- Transfer CLIP training from `meso` repo — check `dev` branch of `ai4bmr-learn` for shared dependencies before porting
- Computation of sample-level scores
- Exploration of sample-level variance
- Computation of cell type / gene-level scores
- Exploration of cell type / gene-level variance
- UMAP embeddings for different models with visualization on WSI
- WSI visualizations of true vs. predicted (tile-level overlay, not UMAP)
- MIL attention scores on tile
