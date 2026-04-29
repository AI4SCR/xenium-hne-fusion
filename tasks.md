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

- Computation of sample-level scores
- Exploration of sample-level variance
- Computation of cell type / gene-level scores
- Exploration of cell type / gene-level variance
- UMAP embeddings for different models with visualization on WSI
- MIL attention scores on tile
