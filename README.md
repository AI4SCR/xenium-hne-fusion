# Xenium x H&E Fusion

Research code for fusing Xenium spatial transcriptomics with H&E whole-slide images.
The main dataset family in this repo is `hest1k`; HESCAPE experiments are built as artifact splits and panels on top of `hest1k`.

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
uv run python scripts/data/run_hest1k.py \
  --config configs/data/local/hest1k.yaml \
  --executor serial
```

This structures raw HEST-1k samples, detects tissues, tiles slides, creates processed tile artifacts, and finalizes `items/all.json` plus processed metadata under `data/03_output/hest1k/`.

### 2. Create HESCAPE lung-healthy artifacts

`configs/artifacts/hescape/lung-healthy.yaml` defines:

- `items.name: hescape/lung-healthy`
- `split.name: hescape/lung-healthy`
- the explicit lung-healthy sample IDs

Run:

```bash
uv run python scripts/artifacts/create_artifacts.py \
  --config configs/artifacts/hescape/lung-healthy.yaml

uv run python scripts/artifacts/create_hescape_splits.py
uv run python scripts/artifacts/create_hescape_panels.py
```

This produces:

- `data/03_output/hest1k/items/hescape/lung-healthy.json`
- `data/03_output/hest1k/splits/hescape/lung-healthy/outer=0-seed=0.parquet`
- `data/03_output/hest1k/panels/hescape/lung-healthy.yaml`
- item stats and overlap figures under `data/03_output/hest1k/figures/`

### 3. Train one model on that split and panel

Example training config:
[configs/train/hescape/expression/lung-healthy/vision.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/train/hescape/expression/lung-healthy/vision.yaml)

Run:

```bash
uv run python scripts/train/supervised.py \
  --config configs/train/hescape/expression/lung-healthy/vision.yaml
```

That config trains on:

- `data.name: hest1k`
- `data.items_path: all.json`
- `data.metadata_path: hescape/lung-healthy/outer=0-seed=0.parquet`
- `data.panel_path: hescape/lung-healthy.yaml`

Optional evaluation config:
[configs/eval/hescape/lung-healthy.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/eval/hescape/lung-healthy.yaml)

## Cluster runs

### Slurm

For dataset creation on a cluster, use the provided wrapper:

```bash
./slurm/run_hest1k.sh --config configs/data/remote/hest1k.yaml
```

It submits one CPU job per sample with `--stage samples`, then one dependent finalization job with `--stage finalize`.

The HESCAPE Slurm experiment reference lives in
[slurm/hescape.md](/Users/adrianomartinelli/projects/xenium-hne-fusion/slurm/hescape.md).

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

The HESCAPE Ray experiment reference lives in
[ray/hescape.md](/Users/adrianomartinelli/projects/xenium-hne-fusion/ray/hescape.md).

## Full HESCAPE experiment suite actually run

The tracked HESCAPE experiment matrix in `slurm/hescape.md` and `ray/hescape.md` covers:

- Organ groups: `breast`, `bowel`, `lung-healthy`, `human-immuno-oncology`, `human-multi-tissue`
- Predefined split folds: `outer=0` through `outer=4` under `data/03_output/hest1k/splits/hescape/<organ>/`
- Base model families: `early-fusion`, `late-fusion-tile`, `late-fusion-token`, `vision`, `expr-tile`, `expr-token`
- Fusion ablation: `--backbone.fusion_strategy concat` for the fusion models
- Gating ablation: `--backbone.learnable_gate true` for the fusion models
- Freeze-morph ablation: `--backbone.freeze_morph true` for `early-fusion`, `late-fusion-tile`, `late-fusion-token`, and `vision`
- Evaluation: `scripts/eval/plot_wandb_scores.py` and `scripts/eval/paired_t_tests.py` with `configs/eval/hescape/<organ>.yaml`

Use the markdown files under `slurm/` and `ray/` as the exact cluster command reference.
