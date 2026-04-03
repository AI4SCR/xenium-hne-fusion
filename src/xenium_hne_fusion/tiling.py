from __future__ import annotations

from pathlib import Path


def detect_tissues(wsi_path: Path, output_parquet: Path) -> None:
    """
    Segment tissue regions in a WSI using HESTTissueSegmentation (DeepLabV3).

    GPU-heavy. Operates at 1–2 µm/px; model weights downloaded automatically on first run.

    Output parquet columns:
        tissue_id (int)
        geometry  (Shapely Polygon, coordinates in WSI pixel space)
    """
    ...


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

    CPU-only. Reads tissue polygons from tissues_parquet and tiles each region
    using lazyslide, resampled to the target mpp.

    Output parquet columns:
        tile_id   (int)
        tissue_id (int)
        geometry  (Shapely Polygon, WSI pixel coords)
        x_px      (int, top-left x in WSI pixel coords)
        y_px      (int, top-left y in WSI pixel coords)
        width_px  (int, always == tile_px after resampling)
        height_px (int, always == tile_px after resampling)

    Output filename convention: {tile_px}_{stride_px}.parquet
    """
    ...
