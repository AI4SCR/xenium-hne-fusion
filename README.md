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

- `items/all.json`: complete tile set used for source-item stats
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

- [configs/artifacts/hest1k.yaml](configs/artifacts/hest1k.yaml)
- [configs/artifacts/hest1k-breast.yaml](configs/artifacts/hest1k-breast.yaml)
- [configs/artifacts/hest1k-lung.yaml](configs/artifacts/hest1k-lung.yaml)
- [configs/artifacts/hest1k-pancreas.yaml](configs/artifacts/hest1k-pancreas.yaml)
- [configs/artifacts/beat.yaml](configs/artifacts/beat.yaml)

`scripts/data/run_hest1k.py`, `scripts/data/run_beat.py`, `scripts/artifacts/filter_items.py`, and `scripts/artifacts/create_splits.py` all use this schema. `filter.include_ids` and `filter.exclude_ids` are mutually exclusive:

```bash
--filter.include_ids '[TENX116]'
--items.name breast
--items.filter.organs '[Breast]'
--split.name breast
--split.group_column_name sample_id
```

## Local CLI

### Manual stages

#### HEST1K structure

```bash
uv run scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml
```

#### BEAT structure

```bash
uv run scripts/data/structure_beat.py beat \
  --config_path configs/data/local/beat.yaml
```

#### Metadata

```bash
uv run python scripts/data/process_metadata.py \
  --config configs/data/local/hest1k.yaml
```

#### HEST1K processing

```bash
uv run python scripts/data/process_hest1k.py \
  --config configs/data/local/hest1k.yaml
```

#### Items

```bash
uv run python scripts/data/create_items.py \
  --config configs/data/local/hest1k.yaml
```

#### Source item stats

Intended to compute summary stats for `items/<items.name>.json` using the artifacts config.
`scripts/artifacts/compute_items_stats.py` parses `ArtifactsConfig`, so it expects an artifacts config.

```bash
uv run scripts/artifacts/compute_items_stats.py \
  --config configs/artifacts/hest1k.yaml
```

#### Filtered items

```bash
uv run scripts/artifacts/filter_items.py \
  --config configs/artifacts/hest1k-breast.yaml
```

#### Splits

```bash
uv run scripts/artifacts/create_splits.py \
  --config configs/artifacts/hest1k-breast.yaml
```

#### Panel

```bash
uv run scripts/artifacts/create_panel.py \
  --config configs/artifacts/hest1k-breast.yaml
```

#### All artifacts

`scripts/artifacts/create_artifacts.py` runs the full artifacts workflow for a single
artifacts config:

1. filter `items/all.json` into `items/<items.name>.json`
2. build `splits/<split.name>/`
3. create or validate `panels/<panel.name>.yaml` when `panel:` is set
4. compute `statistics/all.parquet` for the filtered items

```bash
uv run python scripts/artifacts/create_artifacts.py \
  --config configs/artifacts/hest1k-breast.yaml
```

If the config has no `panel:` section, panel creation is skipped. If `panel:` points to a
predefined panel, the script only checks that the target YAML already exists.

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

#### HEST1K

```bash
uv run scripts/data/run_hest1k.py \
  --config configs/data/remote/hest1k.yaml \
  --name hest1k \
  --executor serial
```

#### BEAT

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

Panel creation is defined in the artifacts config. Keep the output filename in `data.panel_path`, and keep the recipe in a dedicated top-level `panel:` section:

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

`scripts/artifacts/create_panel.py` accepts an artifacts config. It resolves the current artifacts layout, uses the fit split referenced by `split.name` and `panel.metadata_path`, and writes the panel YAML under `data.panel_path`.

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
uv run python scripts/artifacts/create_panel.py \
  --config configs/train/hest1k/expression/breast/early-fusion.yaml \
  --overwrite true

uv run python scripts/artifacts/create_panel.py \
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

### HEST1K

```bash
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml --overwrite
./slurm/train_hest1k_early_fusion_organ.sh breast
```

For organ-specific or thresholded HEST1K runs, reuse the same config and override only what changes:

```bash
uv run python scripts/artifacts/filter_items.py \
  --config configs/artifacts/hest1k-breast.yaml

uv run python scripts/artifacts/create_splits.py \
  --config configs/artifacts/hest1k-breast.yaml
```

### BEAT

```bash
./slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml

sbatch --wrap 'uv run python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml' \
  --job-name beat_early_fusion \
  --partition gpu-l40 \
  --gres gpu:1 \
  --cpus-per-task 12 \
  --mem 64G \
  --time 04:00:00 \
  --output ~/logs/%j.log \
  --error ~/logs/%j.err
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

## Kaiko Ray

Ray helpers live under `ray/`:

```bash
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml"
```

Pass `--entrypoint-num-gpus N` to reserve GPUs for the Ray entrypoint itself.

Useful helpers:

```bash
bash ray/submit.sh
bash ray/scripts/disk_space.sh
bash ray/submit.sh "bash ray/scripts/test_env.sh"
```

### HEST1K

For `hest1k-breast`, keep the data-prep steps separate and then run the bundled artifacts
workflow:

```bash
./ray/submit.sh 'python scripts/data/process_metadata.py --config "configs/data/remote/hest1k.yaml"'
./ray/submit.sh 'python scripts/data/create_items.py --config "configs/data/remote/hest1k.yaml"'
./ray/submit.sh 'python scripts/artifacts/compute_items_stats.py --config "configs/artifacts/hest1k.yaml"'

./ray/submit.sh 'python scripts/artifacts/create_artifacts.py --config "configs/artifacts/hest1k-breast.yaml"'
./ray/submit.sh --entrypoint-num-gpus 1 'python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml'
```

To run the same sequence for other canonical configs:

```bash
export CONFIG=configs/artifacts/hest1k-breast.yaml

./ray/submit.sh "python scripts/data/process_metadata.py --config $CONFIG"
./ray/submit.sh "python scripts/data/create_items.py --config $CONFIG"
./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config $CONFIG"
./ray/submit.sh "python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml"
```

Swap `hest1k-breast.yaml` for `hest1k-lung.yaml` to run the lung config.

### BEAT

```bash
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml"
```

## Development

```bash
uv run pytest
uv add <pkg>
uv add --dev <pkg>
```
## HEST1k Commands

```bash

# data
./ray/submit.sh "python scripts/data/run_hest1k.py --config configs/data/remote/hest1k.yaml --executor ray --stage all --overwrite true"

# artifacts
for ORGAN in breast lung pancreas bowel; do
  
    ./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/hest1k-${ORGAN}.yaml"
    
    for OUTER in 0 1 2 3; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        echo "Creating panel for ${ORGAN} and ${SPLIT_NAME}"

        cmd="python scripts/artifacts/create_panel.py \
            --config configs/artifacts/hest1k-${ORGAN}.yaml \
            --panel.metadata_path ${ORGAN}/${SPLIT_NAME}.parquet \
            --panel.name hvg-${ORGAN}-${ORGAN}-${SPLIT_NAME}"

        echo "Running: ${cmd}"
        ./ray/submit.sh "${cmd}"
    done
done

./ray/submit.sh "python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/scripts/data/create_hescape_splits.py --name hescape-breast --splits_dir splits/hest1k/hescape/human-breast-panel/"
./ray/submit.sh "python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/scripts/data/create_hescape_splits.py --name hescape-lung-healthy --splits_dir splits/hest1k/hescape/human-lung-healthy-panel/"
./ray/submit.sh "python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/scripts/data/create_hescape_splits.py --name hescape-colon --splits_dir splits/hest1k/hescape/human-colon-panel/"

# training
for ORGAN in breast lung pancreas bowel; do
  for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    PANEL_NAME="hvg-${ORGAN}-${ORGAN}-${SPLIT_NAME}.yaml"
    for MODEL in early-fusion late-fusion vision expr-tile expr-token; do
      ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config configs/train/hest1k/expression/${ORGAN}/${MODEL}.yaml --data.metadata_path ${ORGAN}/${SPLIT_NAME}.parquet --data.panel_path ${PANEL_NAME} --debug true"
#      ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config configs/train/hest1k/expression/${ORGAN}/${MODEL}.yaml --data.metadata_path ${ORGAN}/${SPLIT_NAME}.parquet --data.panel_path ${PANEL_NAME}"
    done
  done
done

```

## HEST1k Slurm Commands

```bash
for ORGAN in breast lung pancreas colon; do
    uv run scripts/artifacts/create_artifacts.py --config configs/artifacts/hest1k-${ORGAN}.yaml
    
    for OUTER in 0 1 2 3; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        echo "Creating panel for ${ORGAN} and ${SPLIT_NAME}"

        uv run python scripts/artifacts/create_panel.py \
            --config "configs/artifacts/hest1k-${ORGAN}.yaml" \
            --panel.metadata_path "${ORGAN}/${SPLIT_NAME}.parquet" \
            --panel.name "hvg-${ORGAN}-${ORGAN}-${SPLIT_NAME}"
    done
done
```

## BEAT Commands

```bash
# copy default panel
./ray/submit.sh 'mkdir -p "${DATA_DIR}/03_output/beat/panels/" && cp panels/beat/default.yaml "${DATA_DIR}/03_output/beat/panels/"'
./ray/submit.sh 'cp metadata.bak /raid/ray/shared/fmx/data/processed-v0/datasets/beat/metadata.parquet'

./ray/submit.sh "python scripts/data/create_items.py --config configs/data/remote/beat.yaml"
./ray/submit.sh "python scripts/artifacts/compute_items_stats.py --config configs/artifacts/beat.yaml --items.name=all"  # note: feels a bit hacky
./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat-kaiko.yaml"
./ray/submit.sh "python scripts/artifacts/filter_items.py --config configs/artifacts/beat.yaml"
./ray/submit.sh "python scripts/artifacts/compute_items_stats.py --config configs/artifacts/beat.yaml"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    for MODEL in early-fusion late-fusion vision expr-tile expr-token; do
#      ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config configs/train/beat/expression/${MODEL}.yaml --data.metadata_path default/${SPLIT_NAME}.parquet --debug true"
      ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config configs/train/beat/expression/${MODEL}.yaml --data.metadata_path default/${SPLIT_NAME}.parquet"
    done
done

# gated fusion
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    MODEL=early-fusion
    ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config configs/train/beat/expression/${MODEL}.yaml --data.metadata_path default/${SPLIT_NAME}.parquet --backbone.learnable_gate true --debug true"
#    ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config configs/train/beat/expression/${MODEL}.yaml --data.metadata_path default/${SPLIT_NAME}.parquet"
done

```

## Data Processing Commands

```bash
./ray/submit.sh "python scripts/data/run_beat.py --config configs/data/remote/beat.yaml --executor ray --stage all --overwrite true"
./ray/submit.sh "python scripts/data/run_hest1k.py --config configs/data/remote/hest1k.yaml --executor ray --stage all --overwrite true"
```