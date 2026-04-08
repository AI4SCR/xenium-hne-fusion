Code for the paper [Linking gene expression to morphology with vision-language models in spatial transcriptomics](https://openreview.net/forum?id=h2GcySraTP).

# xenium-hne-fusion

Research codebase for fusing Xenium spatial transcriptomics with H&E whole-slide images.
The current code supports two dataset families:

- `hest1k`: HEST-1k human Xenium data downloaded from Hugging Face
- `beat`: internal BEAT data arranged under a local raw root

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo>
cd xenium-hne-fusion
uv sync
cp .env.example .env
```

`.env` holds machine-specific paths:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=

DATA_DIR=data
HEST1K_RAW_DIR=
BEAT_RAW_DIR=
```

`uv run ...` loads `.env` automatically. In notebooks or ad hoc Python sessions:

```python
from dotenv import load_dotenv

load_dotenv()
```

## Repo structure

```text
xenium-hne-fusion/
├── src/xenium_hne_fusion/      # importable package
├── scripts/data/               # data pipeline entrypoints
├── scripts/train/              # training entrypoints
├── configs/data/               # fully specified dataset processing configs
├── configs/train/              # model/training configs
├── panels/                     # checked-in reference panel artifacts
├── slurm/                      # Slurm submission wrappers
├── ray/                        # Kaiko Ray submission helpers
├── tests/
├── data/                       # managed raw/structured/processed/output data
└── results/                    # local experiment outputs
```

## Managed data layout

All managed paths are derived from `DATA_DIR` plus the dataset `name` in the config.
For `name: hest1k`:

```text
DATA_DIR/01_structured/hest1k/
DATA_DIR/02_processed/hest1k/
DATA_DIR/03_output/hest1k/
```

Typical layout:

```text
data/
├── 00_raw/
│   ├── hest1k/
│   │   ├── HEST_v1_3_0.csv
│   │   ├── wsis/
│   │   └── transcripts/
│   └── beat/
│       ├── metadata.parquet
│       └── <sample_id>/
│           ├── region.tiff
│           └── transcripts/transcripts.parquet
│
├── 01_structured/<name>/
│   ├── metadata.csv|metadata.parquet
│   └── <sample_id>/
│       ├── wsi.tiff
│       ├── transcripts.parquet
│       ├── wsi.png
│       ├── transcripts.png
│       ├── tissues.parquet
│       └── tiles/{tile_px}_{stride_px}.parquet
│
├── 02_processed/<name>/
│   ├── metadata.parquet
│   └── <sample_id>/
│       ├── feature_universe.txt
│       └── {tile_px}_{stride_px}/
│           └── <tile_id>/
│               ├── tile.pt
│               ├── transcripts.parquet
│               ├── expr-kernel_size=16.parquet
│               ├── tile.png
│               ├── transcripts.png
│               └── transcripts_top5_feats.png
│
└── 03_output/<name>/
    ├── items/
    │   ├── all.json
    │   └── <items_name>.json
    ├── statistics/
    │   └── all.parquet
    ├── figures/
    │   └── tile_stats/
    │       └── <items_name>/
    │           └── *.png
    ├── splits/
    │   └── <split_name>/
    │       └── outer=0[-inner=0]-seed=<seed>.parquet
    ├── panels/
    │   └── <panel_name>.yaml
    ├── cache/
    └── logs/
```

Notes:

- `items/all.json` is the complete tile set built from processed tiles.
- `items/<items_name>.json` is a filtered subset, usually driven by `items.filter.*`.
- tile-stat diagnostics live under `DATA_DIR/03_output/<name>/figures/tile_stats/<items_name>/`.
- split parquet files are tile-level tables produced by joining `items/*.json` with sample-level `02_processed/<name>/metadata.parquet`.
- generated panel YAMLs live under `DATA_DIR/03_output/<name>/panels/`.
- fully specified processing configs under `configs/data/` can generate those runtime panel YAMLs.

## Processing config schema

The data pipeline now uses one nested config schema for both datasets:

```yaml
name: hest1k
tiles:
  tile_px: 512
  stride_px: 256
  mpp: 0.5
  kernel_size: 16
  predicate: within
filter:
  species: Homo sapiens
  organ: null
  disease_type: null
  include_ids: null
  exclude_ids: null
items:
  name: default
  filter:
    organs: null
    num_transcripts: 100
    num_unique_transcripts: null
    num_cells: null
    num_unique_cells: null
split:
  name: default
  test_size: 0.25
  val_size: 0.25
  stratify: false
  target_column_name: null
  encode_targets: false
  nan_value: -1
  use_filtered_targets_for_train: false
  include_targets: null
  group_column_name: null
  random_state: 0
panel:
  name: default
  n_top_genes: null
  flavor: null
```

Examples:

- [configs/data/local/hest1k.yaml](configs/data/local/hest1k.yaml)
- [configs/data/local/hest1k-breast.yaml](configs/data/local/hest1k-breast.yaml)
- [configs/data/local/hest1k-lung.yaml](configs/data/local/hest1k-lung.yaml)
- [configs/data/local/hest1k-pancreas.yaml](configs/data/local/hest1k-pancreas.yaml)
- [configs/data/remote/hest1k-breast.yaml](configs/data/remote/hest1k-breast.yaml)
- [configs/data/remote/hest1k-lung.yaml](configs/data/remote/hest1k-lung.yaml)
- [configs/data/remote/hest1k-pancreas.yaml](configs/data/remote/hest1k-pancreas.yaml)
- [configs/data/remote/hest1k.yaml](configs/data/remote/hest1k.yaml)
- [configs/data/local/beat.yaml](configs/data/local/beat.yaml)

`scripts/data/run_hest1k.py`, `scripts/data/run_beat.py`, `scripts/data/filter_items.py`, and `scripts/data/create_splits.py` all use this schema. `filter.include_ids` and `filter.exclude_ids` are mutually exclusive:

```bash
--filter.include_ids '[TENX116]'
--items.name breast
--items.filter.organs '[Breast]'
--split.name breast
--split.group_column_name sample_id
```

## Data pipeline CLI

### Manual stage entrypoints

HEST1K structure and download:

```bash
uv run scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml
```

BEAT structure:

```bash
uv run scripts/data/structure_beat.py beat \
  --config_path configs/data/local/beat.yaml
```

Sample-level metadata cleaning:

```bash
uv run python scripts/data/process_metadata.py \
  --config configs/data/local/hest1k.yaml
```

Per-sample HEST1K processing:

```bash
uv run scripts/data/process_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml \
  --sample_id TENX116
```

Build the base item set:

```bash
uv run python scripts/data/create_items.py \
  --config configs/data/local/hest1k.yaml
```

Compute per-tile filtering statistics:

```bash
uv run scripts/data/compute_tile_stats.py \
  --config configs/data/local/hest1k.yaml
```

Generate a filtered item set:

```bash
uv run scripts/data/filter_items.py \
  --config configs/data/local/hest1k-breast.yaml
```

Create split parquet files from an item set plus sample metadata:

```bash
uv run scripts/data/create_splits.py \
  --config configs/data/local/hest1k-breast.yaml
```

Create a panel from a fully specified organ config:

```bash
uv run scripts/data/create_panel.py \
  --config_path configs/data/local/hest1k-breast.yaml
```

### End-to-end dataset runners

The dataset runners execute the full pipeline in order:

1. structure raw inputs
2. detect tissues
3. tile and process samples
4. write cleaned sample metadata
5. build `items/all.json`
6. compute `statistics/all.parquet`
7. write the filtered item set named by `items.name`
8. write the split collection named by `split.name`

HEST1K:

```bash
uv run scripts/data/run_hest1k.py \
  --config configs/data/remote/hest1k.yaml \
  --name hest1k \
  --executor serial
```

BEAT:

```bash
uv run scripts/data/run_beat.py \
  --config configs/data/remote/beat.yaml \
  --name beat \
  --executor serial
```

Both runners also support `--executor ray` and nested overrides. For example, a single-sample HEST1K run:

```bash
uv run scripts/data/run_hest1k.py \
  --config configs/data/local/hest1k.yaml \
  --name hest1k \
  --executor serial \
  --filter.include_ids '[TENX116]'
```

## Panel recipes and training path resolution

There is one panel concept in the repo:

- `DATA_DIR/03_output/<name>/panels/*.yaml`: runtime panel files consumed by training

Fully specified processing configs under `configs/data/` also carry the panel settings used to generate those runtime panel files.

Training configs under `configs/train/` resolve relative data paths as follows:

- `data.items_path` -> `DATA_DIR/03_output/<name>/items/`
- `data.metadata_path` -> `DATA_DIR/03_output/<name>/splits/`
- `data.panel_path` -> `DATA_DIR/03_output/<name>/panels/`
- `data.cache_dir` -> `DATA_DIR/03_output/<name>/cache/`

If a training config uses absolute paths, they are left unchanged.

Examples:

- BEAT configs live under `configs/train/beat/...`
- HEST1K expression configs are organ-specific and live under `configs/train/hest1k/expression/<organ>/...`

Train a model with:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/beat/expression/early-fusion.yaml
```

The panel YAML loaded by training must contain:

```yaml
source_panel:
  - GENE_A
target_panel:
  - GENE_B
```

## Slurm

The Slurm submission wrappers live in `slurm/`:

```bash
slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml
slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
```

They currently submit one sample per job with:

- `8` CPUs
- `64G` RAM
- `08:00:00` wall time
- logs in `$HOME/logs`

### Full pipeline on Slurm

The Slurm wrappers only submit the per-sample processing stage. To run the full pipeline, use the commands below in order.

HEST1K:

1. Structure and download the samples selected by the data config:

```bash
uv run python scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/remote/hest1k.yaml
```

2. Submit one per-sample processing job per resolved sample:

```bash
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml
```

The wrapped per-sample command is:

```bash
uv run python scripts/data/process_hest1k.py \
  --dataset hest1k \
  --config_path configs/data/remote/hest1k.yaml \
  --sample_id SAMPLE_ID
```

3. After all sample jobs finish, finalize the dataset outputs:

```bash
DATASET_NAME=hest1k
DATASET_NAME=beat

uv run python scripts/data/process_metadata.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/create_items.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/compute_tile_stats.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/filter_items.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/create_splits.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"
```

4. If training needs a panel file, generate it after splits exist:

```bash
uv run python scripts/data/create_panel.py \
  --config_path configs/data/local/hest1k-pancreas.yaml
```

5. Launch training once the data outputs, splits, and panel are ready:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/hest1k/expression/pancreas/early-fusion.yaml
```

Or use the existing HEST1K early-fusion wrapper:

```bash
./slurm/train_hest1k_early_fusion_organ.sh pancreas
```

BEAT:

1. Structure the raw dataset into the canonical layout:

```bash
uv run python scripts/data/structure_beat.py beat \
  --config_path configs/data/remote/beat.yaml
```

2. Submit one per-sample processing job per raw sample directory:

```bash
./slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
```

The wrapper expands each sample job into:

```bash
uv run python scripts/data/detect_tissues.py \
  --wsi_path WSI_PATH \
  --output_parquet TISSUES_PATH

uv run python scripts/data/tile.py \
  --wsi_path WSI_PATH \
  --tissues_parquet TISSUES_PATH \
  --output_parquet TILES_PATH \
  --tile_px TILE_PX \
  --stride_px STRIDE_PX \
  --mpp TILE_MPP

uv run python scripts/data/process.py \
  --wsi_path WSI_PATH \
  --tiles_parquet TILES_PATH \
  --transcripts_path TRANSCRIPTS_PATH \
  --output_dir PROCESSED_DIR \
  --mpp TILE_MPP \
  --predicate PREDICATE \
  --img_size TILE_PX \
  --kernel_size KERNEL_SIZE
```

When `cells.parquet` exists, the wrapper adds:

```bash
--cells_path CELLS_PATH
```

3. After all sample jobs finish, finalize the dataset outputs:

```bash
DATASET_NAME=beat

uv run python scripts/data/process_metadata.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/create_items.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/compute_tile_stats.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/filter_items.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"

uv run python scripts/data/create_splits.py \
  --config "configs/data/remote/$DATASET_NAME.yaml"
```

4. Train once the dataset outputs are ready:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/beat/expression/early-fusion.yaml
```

For organ-specific or thresholded HEST1K runs, keep using the same `--config` file and override only the shared processing fields you need during finalization, for example:

```bash
uv run python scripts/data/filter_items.py \
  --config configs/data/remote/hest1k-breast.yaml

uv run python scripts/data/create_splits.py \
  --config configs/data/remote/hest1k-breast.yaml
```

## Kaiko Ray

Ray submission helpers live under `ray/`. The main entrypoint is:

```bash
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml"
```

For `hest1k-breast`, submit each data-preparation step individually:

```bash
./ray/submit.sh 'python scripts/data/process_metadata.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/create_items.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/compute_tile_stats.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/filter_items.py --config "configs/data/remote/hest1k-breast.yaml" --overwrite=true'
./ray/submit.sh 'python scripts/data/create_splits.py --config "configs/data/remote/hest1k-breast.yaml" --overwrite=true'
./ray/submit.sh 'python scripts/data/create_panel.py --config_path configs/data/remote/hest1k-breast.yaml'
```

Useful helpers:

```bash
bash ray/submit.sh
bash ray/scripts/disk_space.sh
bash ray/submit.sh "bash ray/scripts/test_env.sh"
```

To test the other canonical configs, use the same sequence with the matching config path:

```bash
export CONFIG=configs/data/remote/hest1k-breast.yaml
export CONFIG=configs/data/remote/hest1k-lung.yaml

./ray/submit.sh "python scripts/data/process_metadata.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_items.py --config $CONFIG"
./ray/submit.sh "python scripts/data/filter_items.py --config $CONFIG"
./ray/submit.sh "python scripts/data/compute_tile_stats.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_splits.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_panel.py --config $CONFIG"
```

## Development

```bash
uv run pytest
uv add <pkg>
uv add --dev <pkg>
```
