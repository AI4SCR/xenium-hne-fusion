"""Extract tiles and transcript subsets for a single sample × tile config."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from jsonargparse import auto_cli

from xenium_hne_fusion.processing import (
    extract_tiles,
    load_feature_universe,
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
    cells_path: Path | None = None,
    cell_type_col: str = "Level3_grouped",
    cell_type_universe_path: Path | None = None,
) -> None:
    tiles = gpd.read_parquet(tiles_parquet)
    extract_tiles(wsi_path, tiles, output_dir, mpp)
    tile_transcripts(tiles, transcripts_path, output_dir / "transcripts", predicate)
    process_tiles(tiles, output_dir / "transcripts", output_dir, transcripts_path, img_size, kernel_size)

    if cells_path is not None:
        assert cell_type_universe_path is not None and cell_type_universe_path.exists(), (
            f"--cell_type_universe_path is required when --cells_path is set. Got: {cell_type_universe_path}"
        )
        cell_type_universe = load_feature_universe(cell_type_universe_path)
        tile_cells(tiles, cells_path, output_dir / "cells", predicate)
        process_cell_types(tiles, output_dir / "cells", output_dir, cell_type_universe, cell_type_col, img_size, kernel_size)


if __name__ == "__main__":
    auto_cli(main)
