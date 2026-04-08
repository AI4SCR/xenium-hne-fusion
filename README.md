Code for the paper [Linking gene expression to morphology with vision-language models in spatial transcriptomics](https://openreview.net/forum?id=h2GcySraTP).

# xenium-hne-fusion

Research codebase for fusing Xenium spatial transcriptomics with H&E whole-slide images.
Supports two dataset families:

- `hest1k`: HEST-1k human Xenium data downloaded from Hugging Face
- `beat`: internal BEAT data arranged under a local raw root

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd xenium-hne-fusion
uv sync
cp .env.example .env
```

`.env` holds machine-specific paths and tokens:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=

DATA_DIR=data
HEST1K_RAW_DIR=
BEAT_RAW_DIR=
```

`uv run ...` loads `.env` automatically. For notebooks or ad hoc sessions:

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
├── configs/data/               # processing configs
├── configs/train/              # training configs
├── panels/                     # checked-in panel artifacts
├── slurm/                      # Slurm wrappers
├── ray/                        # Ray submission helpers
├── tests/
├── data/                       # managed raw/structured/processed/output data
└── results/                    # local experiment outputs
```

## Managed data layout

Managed paths are derived from `DATA_DIR` and the dataset `name`.
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

Key outputs:

- `items/all.json`: complete tile set
- `items/<items_name>.json`: filtered subset
- `figures/tile_stats/<items_name>/`: tile-stat plots
- `splits/<split_name>/`: split parquet files
- `panels/`: generated panel YAMLs

## Processing config schema

The data pipeline uses one nested config schema for both datasets:

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

### Manual stages

HEST1K structure:

```bash
uv run scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml
```

BEAT structure:

```bash
uv run scripts/data/structure_beat.py beat \
  --config_path configs/data/local/beat.yaml
```

Metadata:

```bash
uv run python scripts/data/process_metadata.py \
  --config configs/data/local/hest1k.yaml
```

Per-sample processing:

```bash
uv run scripts/data/process_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml \
  --sample_id TENX116
```

Items:

```bash
uv run python scripts/data/create_items.py \
  --config configs/data/local/hest1k.yaml
```

Tile stats:

```bash
uv run scripts/data/compute_tile_stats.py \
  --config configs/data/local/hest1k.yaml
```

Filtered items:

```bash
uv run scripts/data/filter_items.py \
  --config configs/data/local/hest1k-breast.yaml
```

Splits:

```bash
uv run scripts/data/create_splits.py \
  --config configs/data/local/hest1k-breast.yaml
```

Panel:

```bash
uv run scripts/data/create_panel.py \
  --config configs/data/local/hest1k-breast.yaml
```

### End-to-end runners

The dataset runners execute the full pipeline:

1. structure raw inputs
2. detect tissues
3. tile and process samples
4. clean sample metadata
5. build `items/all.json`
6. compute `statistics/all.parquet`
7. write the filtered item set
8. write the split collection

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

Panels are runtime YAML files consumed by training:
- `DATA_DIR/03_output/<name>/panels/*.yaml`

Processing configs under `configs/data/` also carry the panel settings used to generate them.

Training configs under `configs/train/` resolve relative data paths as follows:

- `data.items_path` -> `DATA_DIR/03_output/<name>/items/`
- `data.metadata_path` -> `DATA_DIR/03_output/<name>/splits/`
- `data.panel_path` -> `DATA_DIR/03_output/<name>/panels/`
- `data.cache_dir` -> `DATA_DIR/03_output/<name>/cache/`

If a training config uses absolute paths, they are left unchanged.

Examples:

- BEAT configs live under `configs/train/beat/...`
- HEST1K expression configs are organ-specific and live under `configs/train/hest1k/expression/<organ>/...`

Train a model:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/beat/expression/early-fusion.yaml
```

The panel YAML must contain:

```yaml
source_panel:
  - GENE_A
target_panel:
  - GENE_B
```

## Slurm

Slurm wrappers live in `slurm/`:

```bash
slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml
slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
```

They submit one sample per job with:

- `8` CPUs
- `64G` RAM
- `08:00:00` wall time
- logs in `$HOME/logs`

### Full pipeline on Slurm

The wrappers only submit per-sample processing. Run the rest in order below.

HEST1K:

1. Structure and download the selected samples:

```bash
uv run python scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/remote/hest1k.yaml
```

2. Submit one per-sample job per resolved sample:

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

3. Finalize dataset outputs:

```bash
DATASET_NAME=hest1k

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

4. If training needs a panel, generate it after splits exist:

```bash
uv run python scripts/data/create_panel.py \
  --config configs/data/local/hest1k-pancreas.yaml
```

5. Launch training once data outputs, splits, and panel are ready:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/hest1k/expression/pancreas/early-fusion.yaml
```

Or use the existing HEST1K early-fusion wrapper:

```bash
./slurm/train_hest1k_early_fusion_organ.sh pancreas
```

BEAT:

1. Structure the raw dataset:

```bash
uv run python scripts/data/structure_beat.py beat \
  --config_path configs/data/remote/beat.yaml
```

2. Submit one per-sample job per raw sample directory:

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

3. Finalize dataset outputs:

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

For organ-specific or thresholded HEST1K runs, reuse the same config and override only what changes:

```bash
uv run python scripts/data/filter_items.py \
  --config configs/data/remote/hest1k-breast.yaml

uv run python scripts/data/create_splits.py \
  --config configs/data/remote/hest1k-breast.yaml
```

## Kaiko Ray

Ray helpers live under `ray/`:

```bash
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml"
```

For `hest1k-breast`, submit each step individually:

```bash
./ray/submit.sh 'python scripts/data/process_metadata.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/create_items.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/compute_tile_stats.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/filter_items.py --config "configs/data/remote/hest1k-breast.yaml" --overwrite=true'
./ray/submit.sh 'python scripts/data/create_splits.py --config "configs/data/remote/hest1k-breast.yaml" --overwrite=true'
./ray/submit.sh 'python scripts/data/create_panel.py --config configs/data/remote/hest1k-breast.yaml'
./ray/submit.sh 'python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml'
```

Useful helpers:

```bash
bash ray/submit.sh
bash ray/scripts/disk_space.sh
bash ray/submit.sh "bash ray/scripts/test_env.sh"
```

To run the same sequence for other canonical configs:

```bash
export CONFIG=configs/data/remote/hest1k-breast.yaml

./ray/submit.sh "python scripts/data/process_metadata.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_items.py --config $CONFIG"
./ray/submit.sh "python scripts/data/filter_items.py --config $CONFIG"
./ray/submit.sh "python scripts/data/compute_tile_stats.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_splits.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_panel.py --config $CONFIG"
./ray/submit.sh "python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml"
```

Swap `hest1k-breast.yaml` for `hest1k-lung.yaml` to run the lung config.

## Development

```bash
uv run pytest
uv add <pkg>
uv add --dev <pkg>
```
