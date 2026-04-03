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

Machine-specific paths (data drives, scratch dirs) go in `.env`:

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
├── src/xenium_hne_fusion/          # importable package
│   ├── utils/
│   │   ├── getters.py              # config loading, sample filtering
│   │   └── geometry.py             # coordinate transforms for point geometries
│   ├── download.py                 # HEST download + raw symlinks
│   ├── tiling.py                   # tissue detection (GPU) + tiling (CPU)
│   └── processing.py               # patch extraction + transcript count tensors
│
├── scripts/
│   ├── 0-data-processing/          # data pipeline entry points
│   ├── 1-transcript-prediction/    # train/eval for transcript prediction task
│   └── 2-cell-type-prediction/     # train/eval for cell type prediction task
│
├── tests/
├── data/                           # not tracked — see layout below
├── results/                        # not tracked
└── figures/                        # not tracked
```

## Data layout

```
data/
├── 00_download/hest1k/<sample_id>/      # raw HEST files (HuggingFace snapshot)
│   ├── wsis/                            # pyramidal TIFFs
│   └── transcripts/                     # Xenium transcript parquet
│
├── 01_raw/datasets/hest1k/<sample_id>/  # symlinks into 00_download + derived
│   ├── wsi.tiff                         # symlink → wsis/<file>.tiff
│   ├── transcripts.parquet              # symlink → transcripts/transcripts.parquet
│   └── tiles/
│       └── {tile_px}_{stride_px}.parquet  # GeoDataFrame: tile polygons + pixel bbox
│
└── 02_processed/datasets/hest1k/<sample_id>/{tile_px}_{stride_px}/<tile_id>/
    ├── patch.pt          # uint8 CHW torch tensor  (3 × tile_px × tile_px)
    └── transcripts.pt    # int32 1-D tensor         (n_genes,) — counts per gene
```

> **HPC note**: at full scale (~65 samples × 10k tiles × 2 tile configs) this produces ~2.6M
> files. Check your scratch filesystem inode quota before running at scale.

## Pipeline

The pipeline has four steps, each with a corresponding script and library function.

### 1 — Download

Download HEST-1k samples matching a filter and create raw symlinks.

```bash
uv run scripts/data/download.py --config workflow/config/hest1k.yaml
```

Or interactively:

```python
from xenium_hne_fusion.download import download_sample, create_raw_symlinks
from pathlib import Path

download_sample("TENX95", download_dir=Path("data/00_download/hest1k"))
create_raw_symlinks("TENX95",
    download_dir=Path("data/00_download/hest1k"),
    raw_dir=Path("data/01_raw/datasets/hest1k"))
```

### 2 — Tissue detection  *(GPU)*

Segment tissue regions using HESTTissueSegmentation (DeepLabV3 fine-tuned on HEST-1k).
Model weights are downloaded automatically on first run.

```bash
uv run scripts/data/detect_tissues.py \
    --wsi_path data/01_raw/datasets/hest1k/TENX95/wsi.tiff \
    --output_parquet data/01_raw/datasets/hest1k/TENX95/tissues.parquet
```

Output: GeoDataFrame parquet with `tissue_id` and `geometry` (Shapely Polygons in WSI pixel coords).

### 3 — Tiling  *(CPU)*

Generate a tile grid over detected tissue regions at a target resolution.

```bash
uv run scripts/data/tile.py \
    --wsi_path data/01_raw/datasets/hest1k/TENX95/wsi.tiff \
    --tissues_parquet data/01_raw/datasets/hest1k/TENX95/tissues.parquet \
    --output_parquet data/01_raw/datasets/hest1k/TENX95/tiles/256_256.parquet \
    --tile_px 256 --stride_px 256 --mpp 0.5
```

Output: GeoDataFrame parquet with `tile_id`, `geometry`, `x_px`, `y_px`, `width_px`, `height_px`.

### 4 — Processing

Extract patches and per-tile transcript count tensors.

```bash
uv run scripts/data/process.py \
    --wsi_path data/01_raw/datasets/hest1k/TENX95/wsi.tiff \
    --tiles_parquet data/01_raw/datasets/hest1k/TENX95/tiles/256_256.parquet \
    --transcripts_path data/01_raw/datasets/hest1k/TENX95/transcripts.parquet \
    --output_dir data/02_processed/datasets/hest1k/TENX95/256_256 \
    --mpp 0.5
```

Output per tile: `patch.pt` (uint8 CHW tensor) and `transcripts.pt` (int32 count vector).

## Configuration

Sample filtering is controlled by a YAML config:

```yaml
# workflow/config/hest1k.yaml
metadata_csv: data/00_download/hest1k/HEST_v1_3_0.csv
download_dir: data/00_download/hest1k
raw_dir: data/01_raw/datasets/hest1k
processed_dir: data/02_processed/datasets/hest1k

tile_sizes: [256, 512]
tile_mpp: 0.5

filter:
  species: "Homo sapiens"
  organ: null           # e.g. "breast" or ["breast", "lung"]
  disease_type: null    # e.g. "cancer"
  sample_ids: null      # explicit list overrides all filters above
```

## Development

```bash
uv run pytest           # run tests
uv add <pkg>            # add runtime dependency
uv add --dev <pkg>      # add dev dependency
```
