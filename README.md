Code for the paper [Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer](https://openreview.net/forum?id=h2GcySraTP).

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

HEST1K processing:

```bash
uv run python scripts/data/process_hest1k.py \
  --config configs/data/local/hest1k.yaml
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
  --config configs/train/hest1k/expression/breast/early-fusion.yaml
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

Panel creation is defined in the training config, not the data config. Keep the output filename in `data.panel_path`, and keep the recipe in a dedicated top-level `panel:` section:

```yaml
data:
  name: hest1k
  items_path: breast.json
  metadata_path: breast/outer=0-inner=0-seed=0.parquet
  panel_path: hvg-breast-breast-outer=0-seed=0.yaml

panel:
  name: hvg-breast-breast-outer=0-seed=0
  n_top_genes: 16
  flavor: seurat_v3
```

`scripts/data/create_panel.py` accepts a training config only. It resolves the current training config, uses the fit split referenced by `data.metadata_path`, and writes the panel YAML to `data.panel_path`.

Training configs under `configs/train/` resolve relative data paths as follows:

- `data.items_path` -> `DATA_DIR/03_output/<name>/items/`
- `data.metadata_path` -> `DATA_DIR/03_output/<name>/splits/`
- `data.panel_path` -> `DATA_DIR/03_output/<name>/panels/`
- `data.cache_dir` -> `DATA_DIR/03_output/<name>/cache/`

If a training config uses absolute paths, they are left unchanged.

Examples:

- BEAT configs live under `configs/train/beat/...`
- HEST1K expression configs are organ-specific and live under `configs/train/hest1k/expression/<organ>/...`

Create panels:

```bash
uv run python scripts/data/create_panel.py \
  --config configs/train/hest1k/expression/breast/early-fusion.yaml \
  --overwrite true

uv run python scripts/data/create_panel.py \
  --config configs/train/beat/expression/early-fusion.yaml \
  --overwrite true
```

HEST1K expression configs already define a `panel:` recipe. For BEAT, the command is the same, but the chosen training config must also define a `panel:` recipe if you want `create_panel.py` to generate a panel instead of only checking that `data.panel_path` already exists.

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
slurm/train_hest1k_early_fusion_organ.sh breast
```

The dataset wrappers submit:

- one `samples` job per selected sample
- one `finalize` job that depends on all sample jobs

The wrapped jobs call the existing dataset pipelines in `scripts/data/run_hest1k.py` and `scripts/data/run_beat.py`, so the Slurm layer does not reimplement structuring, tissue detection, tiling, processing, metadata, items, stats, or splits.

Default resources:

- `8` CPUs
- `64G` RAM
- `08:00:00` wall time
- logs in `$HOME/logs`

Pass `--overwrite` to reprocess samples and overwrite finalized outputs.

Examples:

```bash
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml --overwrite
./slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
./slurm/train_hest1k_early_fusion_organ.sh breast
```

The per-sample jobs run:

```bash
uv run python scripts/data/run_<dataset>.py \
  --config CONFIG_PATH \
  --executor serial \
  --stage samples \
  --filter.include_ids "[SAMPLE_ID]" \
  --filter.exclude_ids null
```

The finalization job runs once:

```bash
uv run python scripts/data/run_<dataset>.py \
  --config CONFIG_PATH \
  --executor serial \
  --stage finalize
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

Pass `--entrypoint-num-gpus N` to reserve GPUs for the Ray entrypoint itself.

For `hest1k-breast`, submit each step individually:

```bash
./ray/submit.sh 'python scripts/data/process_metadata.py --config "configs/data/remote/hest1k.yaml"'
./ray/submit.sh 'python scripts/data/create_items.py --config "configs/data/remote/hest1k.yaml"'
./ray/submit.sh 'python scripts/data/compute_tile_stats.py --config "configs/data/remote/hest1k.yaml"'

./ray/submit.sh 'python scripts/data/filter_items.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/compute_tile_stats.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/create_splits.py --config "configs/data/remote/hest1k-breast.yaml"'
./ray/submit.sh 'python scripts/data/create_panel.py --config configs/train/hest1k/expression/breast/early-fusion.yaml'
./ray/submit.sh --entrypoint-num-gpus 1 'python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml'
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
./ray/submit.sh "python scripts/data/create_panel.py --config configs/train/hest1k/expression/breast/early-fusion.yaml"
./ray/submit.sh "python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml"
```

Swap `hest1k-breast.yaml` for `hest1k-lung.yaml` to run the lung config.

## Development

```bash
uv run pytest
uv add <pkg>
uv add --dev <pkg>
```

## Training

```bash
sbatch --wrap 'uv run python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml' --job-name beat_early_fusion --partition gpu-l40 --gres gpu:1 --cpus-per-task 12 --mem 64G --time 04:00:00 --output ~/logs/%j.log --error ~/logs/%j.err
```