from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def get_gene_index(transcripts_path: Path) -> list[str]:
    """
    Return a sorted list of unique gene names from the transcripts parquet.

    Reads only the feature_name column metadata — does not load full file into memory.
    This list defines the gene dimension ordering for all transcripts.pt tensors of a sample.
    """
    ...


def extract_patches(
    wsi_path: Path,
    tiles: gpd.GeoDataFrame,
    output_dir: Path,
    mpp: float,
) -> None:
    """
    Crop and save a patch for each tile in the tile GeoDataFrame.

    For each tile: read region (x_px, y_px, width_px, height_px) from WSI,
    resample to target mpp, convert to uint8 CHW torch tensor.

    Output: <output_dir>/<tile_id>/patch.pt
    """
    ...


def patchify_transcripts(
    tiles: gpd.GeoDataFrame,
    transcripts_path: Path,
    output_dir: Path,
    gene_index: list[str],
    predicate: str = "within",
) -> None:
    """
    Assign transcripts to tiles via chunked spatial join and save per-tile count tensors.

    Processes transcripts in chunks of 1M rows to avoid loading the full file into memory.
    Accumulates per-(tile_id, gene) counts across all chunks, then writes one tensor per tile.

    Output: <output_dir>/<tile_id>/transcripts.pt
        int32 1-D tensor of shape (n_genes,), aligned to gene_index.
        Tiles with zero transcripts receive a zero tensor (file is still written).

    predicate: spatial join predicate passed to gpd.sjoin — 'within' or 'intersects'.
    """
    ...
