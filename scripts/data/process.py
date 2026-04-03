"""Extract patches and transcript count tensors for a single sample × tile config."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from jsonargparse import auto_cli

from xenium_hne_fusion.processing import (
    extract_patches,
    get_gene_index,
    patchify_transcripts,
)


def main(
    wsi_path: Path,
    tiles_parquet: Path,
    transcripts_path: Path,
    output_dir: Path,
    mpp: float = 0.5,
    predicate: str = "within",
) -> None:
    tiles = gpd.read_parquet(tiles_parquet)
    gene_index = get_gene_index(transcripts_path)
    extract_patches(wsi_path, tiles, output_dir, mpp)
    patchify_transcripts(tiles, transcripts_path, output_dir, gene_index, predicate)


if __name__ == "__main__":
    auto_cli(main)
