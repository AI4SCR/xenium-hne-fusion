"""Extract tiles and transcript subsets for a single sample × tile config."""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from jsonargparse import auto_cli
from loguru import logger

from xenium_hne_fusion.processing import (
    extract_tiles,
    process_cell_types,
    process_tiles,
    tile_cells,
    tile_transcripts,
)


def main(
    wsi_path: Path,
    tiles_parquet: Path,
    transcripts_path: Path,
    output_dir: Path,
    mpp: float = 0.5,
    predicate: str = "within",
    img_size: int = 256,
    kernel_size: int = 16,
    cells_path: Optional[Path] = None,
    cell_type_col: str = "Level3_grouped",
) -> None:
    tiles = gpd.read_parquet(tiles_parquet)
    extract_tiles(wsi_path, tiles, output_dir, mpp)
    tile_transcripts(tiles, transcripts_path, output_dir / "transcripts", predicate)
    process_tiles(tiles, output_dir / "transcripts", output_dir, transcripts_path, img_size, kernel_size)

    if cells_path is not None and not cells_path.exists():
        logger.warning(f"cells_path not found, skipping cell type processing: {cells_path}")
    elif cells_path is not None:
        cells = pd.read_parquet(cells_path, columns=[cell_type_col])
        assert hasattr(cells[cell_type_col].dtype, 'categories'), (
            f"{cell_type_col!r} in {cells_path} must be Categorical"
        )
        cell_type_universe = cells[cell_type_col].cat.categories.tolist()
        tile_cells(tiles, cells_path, output_dir / "cells", predicate)
        process_cell_types(tiles, output_dir / "cells", output_dir, cell_type_universe, cell_type_col, img_size, kernel_size)


if __name__ == "__main__":
    auto_cli(main)
