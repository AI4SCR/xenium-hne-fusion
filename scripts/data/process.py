"""Extract tiles and transcript subsets for a single sample × tile config."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from jsonargparse import auto_cli

from xenium_hne_fusion.processing import extract_tiles, process_tiles, tile_transcripts


def main(
    wsi_path: Path,
    tiles_parquet: Path,
    transcripts_path: Path,
    output_dir: Path,
    mpp: float = 0.5,
    predicate: str = "within",
    img_size: int = 256,
    kernel_size: int = 16,
) -> None:
    tiles = gpd.read_parquet(tiles_parquet)
    extract_tiles(wsi_path, tiles, output_dir, mpp)
    tile_transcripts(tiles, transcripts_path, output_dir / "transcripts", predicate)
    process_tiles(tiles, output_dir / "transcripts", output_dir, img_size, kernel_size)


if __name__ == "__main__":
    auto_cli(main)
