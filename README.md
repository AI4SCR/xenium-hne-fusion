Code for the paper [Linking gene expression to morphology with vision-language models in spatial transcriptomics](https://openreview.net/forum?id=h2GcySraTP).

# xenium-hne-fusion

Research codebase for fusing Xenium spatial transcriptomics with H&E whole-slide images.
Built on the [HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest) dataset.

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo>
cd xenium-hne-fusion
uv sync                   # creates .venv and installs all deps
```

Machine-specific paths (raw dataset roots) go in `.env`:

```bash
cp .env.example .env      # edit paths for your machine
```

Then load in scripts or interactive sessions:

```python
from dotenv import load_dotenv; load_dotenv()
```

## Project structure

```
xenium-hne-fusion/
├── src/xenium_hne_fusion/      # importable package
├── panels/                     # generated panel artifacts
├── scripts/data/               # data pipeline entry points
├── scripts/train/              # training entry points
├── tests/
├── data/                       # not tracked — see layout below
├── results/                    # not tracked
└── figures/                    # not tracked
```

## Data layout

```
data/
├── 00_raw/hest1k/                       # raw HEST files (HuggingFace snapshot)
│   ├── HEST_v1_3_0.csv
│   ├── wsis/
│   └── transcripts/
│
├── 01_structured/hest1k/
│   ├── metadata.csv                      # raw metadata symlink
│   └── <sample_id>/                      # canonical per-sample view
│       ├── wsi.tiff                      # symlink → 00_raw/.../wsis/<file>
│       ├── transcripts.parquet           # symlink → 00_raw/.../transcripts/<file>
│       ├── wsi.png                       # sample thumbnail
│       ├── transcripts.png               # 10k streamed transcript overlay
│       ├── tissues.parquet
│       └── tiles/
│           └── {tile_px}_{stride_px}.parquet
│
├── 02_processed/hest1k/
│   ├── metadata.parquet                  # cleaned sample-level metadata
│   └── <sample_id>/{tile_px}_{stride_px}/<tile_id>/
│       ├── tile.pt
│       ├── transcripts.parquet
│       ├── expr-kernel_size=16.parquet
│       ├── tile.png
│       ├── transcripts.png
│       └── transcripts_top5_feats.png
│
└── 03_output/hest1k/                     # dataset-scoped derived outputs
    ├── items/
    │   └── all.json                     # full item set (all complete tiles)
    ├── splits/
    │   └── <split_name>/                # split parquet collection saved via ai4bmr_learn.save_splits
    ├── cache/
    ├── logs/
    └── checkpoints/
```

> **HPC note**: at full scale (~65 samples × 10k tiles × 2 tile configs) this produces ~2.6M
> files. Check your scratch filesystem inode quota before running at scale.

## Pipeline

The pipeline has six steps. Sample metadata stays sample-level in `01_structured` and
`02_processed`, while split metadata becomes tile-level under `03_output`.

### 1 — Download

Download HEST-1k samples matching a dataset config and create structured symlinks.
This also creates:
- `01_structured/<name>/metadata.csv`
- `01_structured/<name>/<sample_id>/wsi.png`
- `01_structured/<name>/<sample_id>/transcripts.png`

```bash
uv run scripts/data/download.py --dataset hest1k
```

Or interactively:

```python
from xenium_hne_fusion.download import (
    create_structured_symlinks,
    download_hest_metadata,
    download_sample,
)
from pathlib import Path

download_hest_metadata(Path("data/00_raw/hest1k"))
download_sample("TENX95", raw_dir=Path("data/00_raw/hest1k"))
create_structured_symlinks("TENX95",
    raw_dir=Path("data/00_raw/hest1k"),
    structured_dir=Path("data/01_structured/hest1k"))
```

### 2 — Metadata cleaning

Convert raw dataset metadata into canonical sample-level parquet:

```bash
uv run scripts/data/process_metadata.py --dataset hest1k
```

Output:
- `data/02_processed/hest1k/metadata.parquet`
- one row per `sample_id`

### 3 — Tissue detection  *(GPU)*

Segment tissue regions using HESTTissueSegmentation (DeepLabV3 fine-tuned on HEST-1k).
Model weights are downloaded automatically on first run.

```bash
uv run scripts/data/detect_tissues.py \
    --wsi_path data/01_structured/hest1k/TENX95/wsi.tiff \
    --output_parquet data/01_structured/hest1k/TENX95/tissues.parquet
```

Output: GeoDataFrame parquet with `tissue_id` and `geometry` (Shapely Polygons in WSI pixel coords).

### 4 — Tiling  *(CPU)*

Generate a tile grid over detected tissue regions at a target resolution.

```bash
uv run scripts/data/tile.py \
    --wsi_path data/01_structured/hest1k/TENX95/wsi.tiff \
    --tissues_parquet data/01_structured/hest1k/TENX95/tissues.parquet \
    --output_parquet data/01_structured/hest1k/TENX95/tiles/256_256.parquet \
    --tile_px 256 --stride_px 256 --mpp 0.5
```

Output: GeoDataFrame parquet with `tile_id`, `geometry`, `x_px`, `y_px`, `width_px`, `height_px`.

### 5 — Processing

Extract tile images, tile-level transcript subsets, and patch-token expression tables.

HEST transcript coordinates are taken only from `he_x` and `he_y`. Some HEST samples
such as `NCBI784` omit raw `geometry`, while samples that do store `geometry` match
`he_x` and `he_y` exactly, so the pipeline rebuilds transcript points from H&E coords
for a single consistent contract.

```bash
uv run scripts/data/process.py \
    --wsi_path data/01_structured/hest1k/TENX95/wsi.tiff \
    --tiles_parquet data/01_structured/hest1k/TENX95/tiles/256_256.parquet \
    --transcripts_path data/01_structured/hest1k/TENX95/transcripts.parquet \
    --output_dir data/02_processed/hest1k/TENX95/256_256 \
    --mpp 0.5
```

Output per tile: `tile.pt`, `transcripts.parquet`, `expr-kernel_size=<k>.parquet`, and QC PNGs.

### 6 — Items and split metadata

Build item records:

```bash
uv run scripts/data/create_items.py --dataset hest1k
```

This creates:
- `03_output/<name>/items/all.json`

Then join items with sample-level metadata and cache tile-level splits:

```bash
uv run scripts/data/create_splits.py --dataset hest1k
```

The split parquet is tile-level. Its index is item `id`, and each tile row keeps:
- tile fields such as `sample_id`, `tile_id`, `tile_dir`
- copied sample-level metadata columns
- the generated `split` column used by `TileDataset`

`create_splits.py` uses `ai4bmr_learn.data.splits.save_splits` and keeps the saved fold set under
`03_output/<name>/splits/<split_name>/`. Training configs and HVG recipes should point to a concrete
parquet inside that folder, such as `default/outer=0-inner=0-seed=0.parquet`.

## Configuration

Sample filtering and dataset naming are controlled by a YAML config:

```yaml
name: hest1k
tile_px: 256
stride_px: 256
tile_mpp: 0.5

filter:
  species: "Homo sapiens"
  organ: null           # e.g. "breast" or ["breast", "lung"]
  disease_type: null    # e.g. "cancer"
  sample_ids: null      # explicit list overrides all filters above
```

`.env` provides the machine-specific roots:

```bash
DATA_DIR=data
HEST1K_RAW_DIR=data/00_raw/hest1k
BEAT_RAW_DIR=/path/to/beat/raw
```

## Training

Training configs stay model-focused. Dataset binding lives under `data.name`, and
relative paths resolve under `DATA_DIR/03_output/<name>/` except for panel artifacts, which live under the repo root `panels/<name>/`.

```yaml
data:
  name: hest1k
  metadata_path: default/outer=0-inner=0-seed=0.parquet
  panel_path: hvg-default-default-outer=0-inner=0-seed=0.yaml
  cache_dir: cache/cell-types       # resolves to DATA_DIR/03_output/hest1k/cache/cell-types
```

`metadata_path` may be absolute. If relative, it is resolved under `03_output/<name>/splits/`.
`panel_path` is resolved relative to `panels/<name>/` and should point to a YAML with `source_panel` and `target_panel`.

`items/all.json` is the unfiltered base item set created by `create_items.py`. Additional item sets such as
`default`, `breast`, `lung`, or `pancreas` are produced separately from `configs/items/<dataset>/`.

Panel-generation recipes live in `configs/panels/<dataset>/<name>.yaml`. They keep `panel_name`
explicit and record the source split parquet via `split_path`, relative to
`DATA_DIR/03_output/<dataset>/splits/`. By convention, the recipe filename matches `panel_name`.

Only BEAT has a predetermined `panels/beat/default.yaml`. HEST1K panel files are expected to use
explicit computed names such as `hvg-default-default-outer=0-inner=0-seed=0.yaml`.

Split recipes live in `configs/splits/<dataset>.yaml`. They are applied on the tile-level
table created by joining `items/all.json` with `02_processed/<name>/metadata.parquet` on `sample_id`.

## Development

```bash
uv run pytest           # run tests
uv add <pkg>            # add runtime dependency
uv add --dev <pkg>      # add dev dependency
```

## Kaiko Ray Submission

Submitting to the Kaiko Ray cluster uses two separate environments:

- a local submission env, `.venv-kray`, which only needs `kray` and related CLI tooling
- a remote job env, generated from `ray/runtime_envs/runtime_env_template.yml` and `ray/runtime_envs/conda.yml`

`ray/submit.sh` is the entrypoint. It reads `.env.kaiko`, renders
`ray/runtime_envs/runtime_env.yml`, uploads the repo as the Ray working directory, and submits
the command through `kray job submit`.

### 1 — Create the local submission env

Prerequisites:

- Kaiko VPN must be enabled during install and job submission.
- `envsubst` must be on `PATH` because `ray/submit.sh` uses it to render the runtime env YAML.
- The Kaiko submission packages come from Kaiko's Nexus Python index.

Create the dedicated submitter env:

```bash
uv venv .venv-kray
source .venv-kray/bin/activate
uv pip install --python .venv-kray/bin/python \
  --extra-index-url https://nexus.infra.prd.kaiko.ai/repository/python-all/simple \
  kaiko-ray-plugins kaiko-kray
```

If `envsubst` is missing on macOS, install `gettext` and expose it on `PATH` before submitting.

### 2 — Verify the repo-side modules used by Ray

The remote runtime uploads local source trees through `py_modules` instead of relying on editable
installs. `ray/submit.sh` expects these paths:

```bash
src/xenium_hne_fusion
ray/other_modules/ai4bmr-learn -> ../../../ai4bmr-learn
```

If the `ai4bmr-learn` symlink is missing, recreate it from the repo root:

```bash
ln -s ../../../ai4bmr-learn ray/other_modules/ai4bmr-learn
```

Do not edit `ray/runtime_envs/runtime_env.yml` by hand. It is generated from the template on each
submission.

### 3 — Create `.env.kaiko`

Cluster-side paths, tokens, and cache locations live in `.env.kaiko`. `ray/submit.sh` exports this
file and injects its values into the Ray runtime env.

Start with something like:

```bash
DATA_DIR=/raid/ray/shared/fmx/data/processed-v0
HEST1K_RAW_DIR=/raid/ray/shared/data/public/bronze/hest
BEAT_RAW_DIR=/raid/ray/shared/data/private

HF_TOKEN=hf_...
WANDB_API_KEY=...
WANDB_BASE_URL=https://api.wandb.ai

HF_HOME=/raid/ray/shared/cache/huggingface
HF_HUB_CACHE=/raid/ray/shared/cache/huggingface/hub
TORCH_HOME=/raid/ray/shared/cache/torch
DATASETS_CACHE=/raid/ray/shared/cache/datasets
CACHE_DIR_HELICAL=/raid/ray/shared/cache/helical
```

`WANDB_API_KEY` is required by the submission wrapper. The dataset and cache paths above are the
ones typically needed by real data-processing and training jobs.

### 4 — Smoke test the submission path

First activate the local submitter env:

```bash
source .venv-kray/bin/activate
```

Then run a few small jobs in order:

```bash
bash ray/submit.sh
bash ray/scripts/disk_space.sh
bash ray/submit.sh "bash ray/scripts/test_env.sh"
```

These check, respectively:

- the basic `pwd` submission path
- filesystem visibility inside the cluster job
- that `xenium_hne_fusion`, `ai4bmr_learn`, `torch`, and `lightning` import correctly in the remote env

### 5 — Submit repo scripts

The general pattern is:

```bash
bash ray/submit.sh "python <repo-script> <args>"
```

Examples:

```bash
bash ray/submit.sh "python scripts/data/process_metadata.py --dataset hest1k"
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/hest1k/expression/early-fusion.yaml"
```

For the end-to-end HEST1K pipeline there is also a small wrapper that forwards its arguments to
`scripts/data/run_hest1k.py` before submitting:

```bash
bash ray/scripts/run_hest1k.sh --sample_id TENX95
bash ray/scripts/run_hest1k.sh --organ breast
```

When adding a new cluster job, prefer testing the command locally with `uv run ...` first, then
submit the same Python invocation through `bash ray/submit.sh "python ..."`.
