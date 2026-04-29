# Xenium x H&E Fusion

Research code for ["Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer"](https://openreview.net/forum?id=h2GcySraTP) (ICLR 2026 Workshop LMRL) — an early-fusion multimodal transformer that integrates subcellular Xenium transcript readouts directly into the ViT token stream to enable fine-grained cross-modal interaction for gene expression prediction.

Primary results are on an internal Xenium cohort (BEAT); we also benchmark on [`hest1k`](https://arxiv.org/abs/2406.16192) using splits and panels from the [HESCAPE](https://arxiv.org/abs/2508.01490) benchmark.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{martinelli2026,
  title     = {Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer},
  author    = {Adriano Martinelli and Bernd Illing and Isinsu Katircioglu and Alice Driessen and Fei Tang and Robert Berke and Raphael Gottardo and Marianna Rapsomaniki},
  booktitle = {ICLR 2026 Workshop on Learning and Mining with Representation Learning (LMRL)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=h2GcySraTP},
}
```

## Project structure

```text
xenium-hne-fusion/
├── src/xenium_hne_fusion/   # reusable package code
├── scripts/data/            # dataset structuring and processing entrypoints
├── scripts/artifacts/       # items, splits, panels, and stats
├── scripts/train/           # training entrypoints
├── scripts/eval/            # W&B plots and paired tests
├── configs/data/            # dataset processing configs
├── configs/artifacts/       # artifact generation configs
├── configs/train/           # training configs
├── configs/eval/            # evaluation configs
├── slurm/                   # Slurm wrappers and experiment command refs
├── ray/                     # Ray submission helpers and command refs
├── tests/                   # pytest suite
├── data/                    # managed raw / structured / processed / output data
└── results/                 # local outputs, not managed dataset artifacts
```

## Managed data layout

All managed paths are derived from `DATA_DIR` and the dataset `name`.
For `name: hest1k`, the pipeline uses:

```text
DATA_DIR/00_raw/hest1k/
DATA_DIR/01_structured/hest1k/
DATA_DIR/02_processed/hest1k/
DATA_DIR/03_output/hest1k/
```

The important outputs are:

```text
data/
├── 00_raw/hest1k/
│   ├── HEST_v1_3_0.csv
│   ├── wsis/
│   └── transcripts/
├── 01_structured/hest1k/
│   ├── metadata.csv|metadata.parquet
│   └── <sample_id>/
│       ├── wsi.tiff
│       ├── transcripts.parquet
│       ├── wsi.png
│       ├── transcripts.png
│       ├── tissues.parquet
│       └── tiles/<tile_px>_<stride_px>.parquet
├── 02_processed/hest1k/
│   ├── metadata.parquet
│   └── <sample_id>/<tile_px>_<stride_px>/<tile_id>/
│       ├── tile.pt
│       ├── transcripts.parquet
│       └── expr-kernel_size=16.parquet
└── 03_output/hest1k/
    ├── items/all.json
    ├── items/<items_name>.json
    ├── statistics/*.parquet
    ├── splits/<split_name>/*.parquet
    ├── panels/<panel_name>.yaml
    ├── figures/
    └── cache/
```

`items/all.json` is the full tile inventory. Artifact configs filter that into task-specific item sets, create split metadata, and create or validate gene panels under `03_output/hest1k/panels/`.

## Environment with `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd xenium-hne-fusion
uv sync
cp .env.example .env
```

Use `uv run ...` for all entrypoints. It loads `.env` automatically.

```bash
uv run pytest
uv run python scripts/data/run_hest1k.py --config configs/data/local/hest1k.yaml --executor serial
uv add <pkg>
uv add --dev <pkg>
```

For notebooks or ad hoc Python sessions:

```python
from dotenv import load_dotenv

load_dotenv()
```

## `.env` setup

`.env.example` contains the required machine-specific values:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=

# SLURM
ACCOUNT=

DATA_DIR=data
HEST1K_RAW_DIR=
BEAT_RAW_DIR=
```

What each variable does:

- `HF_TOKEN`: required to download gated HEST-1k assets.
- `WANDB_API_KEY`: needed for tracked training and evaluation plots.
- `ACCOUNT`: used by cluster-specific Slurm wrappers.
- `DATA_DIR`: root for `00_raw`, `01_structured`, `02_processed`, and `03_output`.
- `HEST1K_RAW_DIR`: raw HEST-1k location.
- `BEAT_RAW_DIR`: raw BEAT location if you use that dataset.

## Config files

The repo is driven by YAML configs in four layers.

- `configs/data/{local,remote}/hest1k.yaml`: dataset processing config. Defines `name`, tile settings, and sample filters. `local` is for direct workstation runs; `remote` is the broad cluster version.
- `configs/artifacts/hescape/lung-healthy.yaml`: artifact config. Defines the HESCAPE item subset and split name for the lung-healthy panel.
- `configs/train/hescape/expression/lung-healthy/*.yaml`: training configs. One file per model family.
- `configs/eval/hescape/lung-healthy.yaml`: evaluation config for W&B score plots and paired tests.

The key training paths stay relative to `DATA_DIR/03_output/<name>/`:

- `data.items_path`
- `data.metadata_path`
- `data.panel_path`
- `data.cache_dir`

## One concrete example: HESCAPE `lung-healthy`

This is the simplest end-to-end path to keep in mind.

### 1. Create the base `hest1k` data

```bash
# Download missing raw HEST-1k files and build the structured dataset layout.
uv run python scripts/data/structure_hest1k.py \
  --config configs/data/local/hest1k.yaml

# Detect tissues, tile each slide, and write per-tile processed artifacts.
uv run python scripts/data/process_hest1k.py \
  --config configs/data/local/hest1k.yaml

# Normalize sample metadata into the canonical processed parquet.
uv run python scripts/data/process_metadata.py \
  --config configs/data/local/hest1k.yaml

# Build the tile-level training item list at data/03_output/hest1k/items/all.json.
uv run python scripts/data/create_items.py \
  --config configs/data/local/hest1k.yaml

# Compute dataset-wide item statistics and summary plots.
uv run python scripts/data/compute_all_items_stats.py \
  --config configs/data/local/hest1k.yaml
```

This makes the pipeline stages explicit:

- `structure_hest1k.py`: downloads missing HEST-1k samples, validates MPP, and builds `data/01_structured/hest1k/`
- `process_hest1k.py`: detects tissues, tiles slides, extracts tile crops, and writes per-tile processed artifacts under `data/02_processed/hest1k/`
- `process_metadata.py`: writes cleaned sample metadata to `data/02_processed/hest1k/metadata.parquet`
- `create_items.py`: builds `data/03_output/hest1k/items/all.json`
- `compute_all_items_stats.py`: writes the base item statistics used by downstream artifact steps

### 2. Create HESCAPE lung-healthy artifacts

`configs/artifacts/hescape/lung-healthy.yaml` defines:

- `items.name: hescape/lung-healthy`
- `split.name: hescape/lung-healthy`
- the explicit lung-healthy sample IDs

Run:

```bash
# Filter all.json down to the lung-healthy HESCAPE subset.
uv run python scripts/artifacts/filter_items.py \
  --config configs/artifacts/hescape/lung-healthy.yaml

# Materialize the fixed HESCAPE outer-fold split parquet files.
uv run python scripts/artifacts/create_hescape_splits.py

# Build the HESCAPE source/target panel YAML from the split feature universe.
uv run python scripts/artifacts/create_hescape_panels.py

# Compute subset-specific item statistics and plots.
uv run python scripts/artifacts/compute_items_stats.py \
  --config configs/artifacts/hescape/lung-healthy.yaml
```

This produces:

- `data/03_output/hest1k/items/hescape/lung-healthy.json`
- `data/03_output/hest1k/splits/hescape/lung-healthy/outer=0-seed=0.parquet`
- `data/03_output/hest1k/panels/hescape/lung-healthy.yaml`
- item stats under `data/03_output/hest1k/statistics/` and item-stat figures under `data/03_output/hest1k/figures/items/stats/`

### 3. Train one model on that split and panel

Example training config:
[configs/train/hescape/expression/lung-healthy/vision.yaml](configs/train/hescape/expression/lung-healthy/vision.yaml)

Run:

```bash
# Train the vision-only baseline on the lung-healthy split and panel.
uv run python scripts/train/supervised.py \
  --config configs/train/hescape/expression/lung-healthy/vision.yaml
```

That config trains on:

- `data.name: hest1k`
- `data.items_path: all.json`
- `data.metadata_path: hescape/lung-healthy/outer=0-seed=0.parquet`
- `data.panel_path: hescape/lung-healthy.yaml`

Optional evaluation config:
[configs/eval/hescape/lung-healthy.yaml](configs/eval/hescape/lung-healthy.yaml)

## Cluster runs

### Slurm

For dataset creation on a cluster, use the provided wrapper:

```bash
./slurm/run_hest1k.sh --config configs/data/remote/hest1k.yaml
```

It submits one CPU job per sample with `--stage samples`, then one dependent finalization job with `--stage finalize`.

The HESCAPE Slurm experiment reference lives in [slurm/hescape.md](slurm/hescape.md).
Dataset submission commands for HEST-1k and BEAT are in [slurm/hest1k.md](slurm/hest1k.md) and [slurm/beat.md](slurm/beat.md).

### Ray

For Ray data processing:

```bash
./ray/scripts/run_data.sh --config configs/data/remote/hest1k.yaml
```

For Ray training:

```bash
./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
  "python scripts/train/supervised.py --config configs/train/hescape/expression/lung-healthy/vision.yaml"
```

The HESCAPE Ray experiment reference lives in [ray/hescape.md](ray/hescape.md).

## Task tracking

Current experiment status and next tasks live in
[tasks.md](/Users/adrianomartinelli/projects/xenium-hne-fusion/tasks.md).

Use the markdown files under `slurm/` and `ray/` as the exact cluster command reference.
