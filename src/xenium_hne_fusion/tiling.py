
from pathlib import Path

import geopandas as gpd
import lazyslide as zs
import numpy as np
from loguru import logger
from spatialdata.models import ShapesModel
from wsidata import open_wsi
import matplotlib.pyplot as plt


def detect_tissues(wsi_path: Path, output_parquet: Path) -> None:
    """
    Segment tissue regions using lazyslide threshold-based detection.

    Output parquet columns:
        tissue_id (int), geometry (Shapely Polygon, WSI pixel coords)
    """
    logger.info(f"Detecting tissues: {wsi_path.name}")
    wsi = open_wsi(wsi_path)
    zs.pp.find_tissues(wsi)

    _, ax = plt.subplots()
    zs.pl.tissue(wsi, ax=ax)
    # ax.figure.show()
    ax.figure.savefig(output_parquet.with_suffix('.png'))
    plt.close('all')

    tissues: gpd.GeoDataFrame = wsi["tissues"]
    logger.info(f"Found {len(tissues)} tissue region(s)")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    tissues.to_parquet(output_parquet)


def tile_tissues(
    wsi_path: Path,
    tissues_parquet: Path,
    tile_px: int,
    stride_px: int,
    mpp: float,
    output_parquet: Path,
    slide_mpp: float | None = None,
) -> None:
    """
    Generate a tile grid over detected tissue regions.

    CPU-only. Output parquet columns:
        tile_id, tissue_id, geometry (WSI pixel coords),
        x_px, y_px, width_px, height_px
    """
    logger.info(
        f"Tiling {wsi_path.name} — tile_px={tile_px}, stride_px={stride_px}, mpp={mpp}, slide_mpp={slide_mpp}"
    )
    wsi = open_wsi(wsi_path)
    wsi["tissues"] = ShapesModel.parse(gpd.read_parquet(tissues_parquet))
    zs.pp.tile_tissues(wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp, slide_mpp=slide_mpp)

    _, ax = plt.subplots()
    zs.pl.tiles(wsi, ax=ax)
    # ax.figure.show()
    ax.figure.savefig(output_parquet.with_suffix('.png'))
    plt.close('all')

    tiles: gpd.GeoDataFrame = wsi["tiles"].copy()
    bounds = tiles.geometry.bounds  # minx, miny, maxx, maxy
    tiles["x_px"] = bounds["minx"].astype(int)
    tiles["y_px"] = bounds["miny"].astype(int)
    tiles["width_px"] = (bounds["maxx"] - bounds["minx"]).astype(int)
    tiles["height_px"] = (bounds["maxy"] - bounds["miny"]).astype(int)

    logger.info(f"Generated {len(tiles)} tiles")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    tiles.to_parquet(output_parquet)


def save_wsi_thumbnail(wsi_path: Path, output_path: Path, max_size: int = 2048) -> None:
    """Save a downsampled WSI thumbnail as PNG for quick inspection."""
    from PIL import Image

    wsi = open_wsi(wsi_path)
    arr = wsi.reader.get_thumbnail(max_size)  # (H, W, 3) uint8
    h, w = arr.shape[:2]
    logger.info(f"Thumbnail size: {w}×{h}")
    img = Image.fromarray(arr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Thumbnail saved to {output_path}")


def save_points_overview(
    wsi_path: Path,
    points_path: Path,
    output_path: Path,
    n: int = 10_000,
    max_size: int = 2048,
    seed: int = 0,
    label: str | None = None,
) -> None:
    """Plot n random points on the WSI thumbnail. Row-group-scoped sampling to control memory.

    Requires a `geometry` column; datasets with raw `he_x`/`he_y` coordinates must be
    normalized to `geometry` upstream (see `scripts/data/hest1k/process.py`).
    """
    import openslide
    import pyarrow.dataset as ds

    from ai4bmr_learn.plotting.xenium import visualize_points
    from PIL import Image

    label = label or points_path.stem
    dataset = ds.dataset(points_path, format="parquet")
    assert "geometry" in dataset.schema.names, f"Missing geometry column: {points_path}"
    total_rows = dataset.count_rows()  # metadata only, no data read
    n = min(n, total_rows)
    logger.info(f"Sampling {n} {label} from {total_rows} total rows")

    idx = np.random.default_rng(seed).choice(total_rows, size=n, replace=False)
    table = dataset.take(sorted(idx), columns=["geometry"])  # reads only row groups containing idx
    points = gpd.GeoDataFrame.from_arrow(table)
    logger.info(f"Collected {len(points)} {label} for overlay")

    slide = openslide.OpenSlide(str(wsi_path))
    viz = visualize_points(points, slide=slide, num_points=None, max_size=max_size, radius=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viz).save(output_path)
    logger.info(f"{label.capitalize()} overview saved to {output_path}")


