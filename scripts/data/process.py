"""Extract tiles and transcript subsets for a single sample × tile config."""

from pathlib import Path
from typing import Optional

import geopandas as gpd
from jsonargparse import auto_cli
from loguru import logger

from xenium_hne_fusion.processing import (
    extract_tiles,
    process_cells,
    process_tiles,
    tile_cells,
    tile_transcripts,
)
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL


def main(
    wsi_path: Path,
    tiles_parquet: Path,
    transcripts_path: Path,
    output_dir: Path,
    mpp: float = 0.5,
    native_mpp: Optional[float] = None,
    predicate: str = "within",
    *,
    img_size: int,
    kernel_size: int = 16,
    cells_path: Optional[Path] = None,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
) -> None:
    tiles = gpd.read_parquet(tiles_parquet)
    extract_tiles(wsi_path, tiles, output_dir, mpp, native_mpp=native_mpp, img_size=img_size)
    tile_transcripts(tiles, transcripts_path, output_dir, img_size=img_size, predicate=predicate)
    process_tiles(tiles, output_dir, img_size=img_size, kernel_size=kernel_size)

    if cells_path is not None and not cells_path.exists():
        logger.warning(f"cells_path not found, skipping cell processing: {cells_path}")
    elif cells_path is not None:
        tile_cells(tiles, cells_path, output_dir, predicate=predicate)
        process_cells(tiles, output_dir, img_size, cell_type_col=cell_type_col)


def cli() -> int:
    auto_cli(main)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
