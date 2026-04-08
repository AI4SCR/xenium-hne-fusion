
from pathlib import Path

import geopandas as gpd
import lazyslide as zs
import pandas as pd
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


def _get_transcript_coordinate_columns(schema_names: list[str]) -> list[str]:
    if {"he_x", "he_y"} <= set(schema_names):
        return ["he_x", "he_y"]
    assert "geometry" in schema_names, f"Missing transcript coordinates: {schema_names}"
    return ["geometry"]


def _load_transcript_batch(batch) -> gpd.GeoDataFrame:
    schema_names = set(batch.schema.names)
    if {"he_x", "he_y"} <= schema_names:
        chunk = batch.to_pandas()
        chunk["geometry"] = gpd.points_from_xy(chunk["he_x"], chunk["he_y"])
        return gpd.GeoDataFrame(chunk, geometry="geometry")

    assert "geometry" in schema_names, f"Missing transcript coordinates: {batch.schema.names}"
    return gpd.GeoDataFrame.from_arrow(batch)


def save_transcript_overview(
    wsi_path: Path,
    transcripts_path: Path,
    output_path: Path,
    n: int = 10_000,
    max_size: int = 2048,
    seed: int = 0,
) -> None:
    """Plot n random transcripts on the WSI thumbnail. Stream-sample from parquet to control memory."""
    import openslide
    import pyarrow.parquet as pq

    from ai4bmr_learn.plotting.xenium import visualize_points
    from PIL import Image

    rng = np.random.default_rng(seed)

    # --- stream-sample transcripts ---
    pf = pq.ParquetFile(transcripts_path)
    total_rows = pf.metadata.num_rows
    batch_size = 65_536
    num_batches = max(1, (total_rows + batch_size - 1) // batch_size)
    logger.info(f"Sampling {n} transcripts from {total_rows} total rows across {num_batches} batches")

    n = min(n, total_rows)
    columns = _get_transcript_coordinate_columns(pf.schema_arrow.names)
    collected: list[gpd.GeoDataFrame] = []
    taken_total = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), start=1):
        batches_left = num_batches - batch_idx + 1
        needed = n - taken_total
        if needed <= 0:
            continue
        num_take = max(1, int(np.ceil(needed / batches_left)))
        size = len(batch)
        if size <= num_take:
            collected.append(_load_transcript_batch(batch))
            taken_total += size
        else:
            idx = rng.choice(size, size=num_take, replace=False)
            collected.append(_load_transcript_batch(batch.take(idx)))
            taken_total += num_take

    points = pd.concat(collected, ignore_index=True)
    points = gpd.GeoDataFrame(points, geometry="geometry")
    logger.info(f"Collected {len(points)} transcripts for overlay")

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        viz = visualize_points(points, slide=slide, num_points=None, max_size=max_size, radius=1)
    finally:
        slide.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viz).save(output_path)
    logger.info(f"Transcript overview saved to {output_path}")


def save_sample_overview(
    wsi_path: Path,
    transcripts_path: Path,
    output_dir: Path,
    n: int = 10_000,
    max_size: int = 2048,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_wsi_thumbnail(wsi_path, output_dir / "wsi.png", max_size=max_size)
    save_transcript_overview(
        wsi_path,
        transcripts_path,
        output_dir / "transcripts.png",
        n=n,
        max_size=max_size,
    )
