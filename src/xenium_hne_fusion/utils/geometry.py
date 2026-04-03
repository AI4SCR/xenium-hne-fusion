from __future__ import annotations

import numpy as np
import geopandas as gpd


def to_origin(points: gpd.GeoDataFrame, tile: gpd.GeoSeries) -> gpd.GeoDataFrame:
    """Shift point coordinates so tile top-left corner becomes (0, 0)."""
    ...


def transform_points(
    points: gpd.GeoDataFrame,
    tile: gpd.GeoSeries,
    dst_height: int,
    dst_width: int,
    subset: np.ndarray | None = None,
    errors: str = "raise",
) -> gpd.GeoDataFrame:
    """
    Shift and scale point geometries from WSI pixel coords to target image pixel coords.

    subset: boolean mask selecting which rows of points to transform.
            If None, all rows are transformed.
    errors: 'raise' | 'warn' | 'clip' — controls out-of-bounds handling.
            Values can be combined, e.g. 'warn+clip'.
    """
    ...


def validate_points(
    xs: np.ndarray,
    ys: np.ndarray,
    height: int,
    width: int,
    errors: str = "raise",
) -> None:
    """
    Assert all points fall within [0, width] × [0, height].

    Tolerates floating-point boundary cases within atol=1e-6.
    errors: 'raise' | 'warn' | 'clip'.
    """
    ...
