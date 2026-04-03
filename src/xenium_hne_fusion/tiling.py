from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import lazyslide as zs
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
