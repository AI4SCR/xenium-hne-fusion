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
    │   ├── <split_name>.parquet         # canonical tile-level metadata for training
    │   └── <split_name>/                # full split set saved via ai4bmr_learn.save_splits
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

`create_splits.py` uses `ai4bmr_learn.data.splits.save_splits`, keeps the full saved fold set under
`03_output/<name>/splits/<split_name>/`, and copies the canonical `outer=0-inner=0` split to
`03_output/<name>/splits/<split_name>.parquet` for direct training use.

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
  metadata_path: splits/default.parquet  # tile-level split metadata
  panel_path: default.yaml          # resolves to panels/hest1k/default.yaml
  cache_dir: cache/cell-types       # resolves to DATA_DIR/03_output/hest1k/cache/cell-types
```

`metadata_path` should point to the canonical parquet at `03_output/<name>/splits/<split_name>.parquet`.
`panel_path` is resolved relative to `panels/<name>/` and should point to a YAML with `source_panel` and `target_panel`.

Panel-generation recipes live in `configs/panels/<dataset>/<name>.yaml`.

Split recipes live in `configs/splits/<dataset>.yaml`. They are applied on the tile-level
table created by joining `items/all.json` with `02_processed/<name>/metadata.parquet` on `sample_id`.

## Development

```bash
uv run pytest           # run tests
uv add <pkg>            # add runtime dependency
uv add --dev <pkg>      # add dev dependency
```
