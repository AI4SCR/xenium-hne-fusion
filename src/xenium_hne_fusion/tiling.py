from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import lazyslide as zs
import numpy as np
from loguru import logger
from spatialdata.models import ShapesModel
from wsidata import open_wsi


def detect_tissues(wsi_path: Path, output_parquet: Path) -> None:
    """
    Segment tissue regions using lazyslide threshold-based detection.

    Output parquet columns:
        tissue_id (int), geometry (Shapely Polygon, WSI pixel coords)
    """
    logger.info(f"Detecting tissues: {wsi_path.name}")
    wsi = open_wsi(wsi_path)
    zs.pp.find_tissues(wsi)
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
) -> None:
    """
    Generate a tile grid over detected tissue regions.

    CPU-only. Output parquet columns:
        tile_id, tissue_id, geometry (WSI pixel coords),
        x_px, y_px, width_px, height_px
    """
    logger.info(f"Tiling {wsi_path.name} — tile_px={tile_px}, stride_px={stride_px}, mpp={mpp}")
    wsi = open_wsi(wsi_path)
    wsi["tissues"] = ShapesModel.parse(gpd.read_parquet(tissues_parquet))
    zs.pp.tile_tissues(wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp)

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


def save_transcript_overview(
    wsi_path: Path,
    transcripts_path: Path,
    output_path: Path,
    n: int = 10_000,
    max_size: int = 2048,
    seed: int = 0,
) -> None:
    """Plot n random transcripts on the WSI thumbnail. Stream-sample from parquet to control memory."""
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    from PIL import Image

    rng = np.random.default_rng(seed)

    # --- build thumbnail ---
    wsi = open_wsi(wsi_path)
    arr = wsi.reader.get_thumbnail(max_size)  # (H, W, 3) uint8
    thumb_h, thumb_w = arr.shape[:2]
    props = wsi.reader.properties
    wsi_h, wsi_w = props.level_shape[0]  # full-res (height, width)
    scale_x = thumb_w / wsi_w
    scale_y = thumb_h / wsi_h

    logger.info(f"Thumbnail size: {thumb_w}×{thumb_h}")
    thumb = Image.fromarray(arr)

    # --- stream-sample transcripts ---
    pf = pq.ParquetFile(transcripts_path)
    row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    logger.info(f"Sampling {n} transcripts from {total_rows} total rows across {row_groups} row groups")

    n = min(n, total_rows)
    collected: list = []
    n_collected = 0

    order = rng.permutation(row_groups)
    for rg_idx in order:
        if n_collected >= n:
            break
        needed = n - n_collected
        table = pf.read_row_group(rg_idx, columns=["he_x", "he_y"])
        size = len(table)
        if size <= needed:
            collected.append(table.to_pydict())
            n_collected += size
        else:
            idx = rng.choice(size, size=needed, replace=False)
            collected.append(table.take(idx).to_pydict())
            n_collected += needed

    xs = np.concatenate([np.asarray(d["he_x"]) for d in collected])
    ys = np.concatenate([np.asarray(d["he_y"]) for d in collected])
    xs = xs * scale_x
    ys = ys * scale_y
    logger.info(f"Collected {len(xs)} transcripts for overlay")

    # --- plot ---
    dpi = 150
    fig, ax = plt.subplots(figsize=(thumb_w / dpi, thumb_h / dpi), dpi=dpi)
    ax.imshow(thumb)
    ax.scatter(xs, ys, s=0.5, c="red", linewidths=0, alpha=0.4, rasterized=True)
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    logger.info(f"Transcript overview saved to {output_path}")
