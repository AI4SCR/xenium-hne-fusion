from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from loguru import logger
from wsidata import open_wsi


def get_gene_index(transcripts_path: Path) -> list[str]:
    """
    Return sorted unique gene names from transcripts parquet without loading full file.
    Defines the gene dimension for all transcripts.pt tensors of a sample.
    """
    pf = pq.ParquetFile(transcripts_path)
    genes = set()
    for batch in pf.iter_batches(columns=["feature_name"]):
        genes.update(batch.column("feature_name").to_pylist())
    return sorted(genes)


def extract_patches(
    wsi_path: Path,
    tiles: gpd.GeoDataFrame,
    output_dir: Path,
    mpp: float,
) -> None:
    """
    Crop and save a patch.pt for each tile.

    Reads native-resolution region, resizes to tile_px × tile_px at target mpp,
    saves as uint8 CHW torch tensor.
    """
    from PIL import Image

    wsi = open_wsi(wsi_path)
    native_mpp = wsi.properties.mpp
    assert native_mpp is not None, "WSI has no mpp metadata"

    logger.info(f"Extracting {len(tiles)} patches (native mpp={native_mpp:.4f}, target mpp={mpp})")
    for _, tile in tiles.iterrows():
        tile_dir = output_dir / str(tile.tile_id)
        tile_dir.mkdir(parents=True, exist_ok=True)

        x, y, w, h = tile.x_px, tile.y_px, tile.width_px, tile.height_px
        tile_px = round(w * native_mpp / mpp)

        img = wsi.reader.get_region(x, y, w, h, level=0)  # (H, W, 3) uint8
        img = Image.fromarray(img).resize((tile_px, tile_px), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # CHW
        torch.save(tensor, tile_dir / "patch.pt")

    logger.info("Patch extraction done")


def patchify_transcripts(
    tiles: gpd.GeoDataFrame,
    transcripts_path: Path,
    output_dir: Path,
    gene_index: list[str],
    predicate: str = "within",
) -> None:
    """
    Assign transcripts to tiles via chunked spatial join, save per-tile count tensors.

    Transcripts are joined using he_x/he_y (H&E pixel coords).
    Output: <output_dir>/<tile_id>/transcripts.pt — int32 tensor of shape (n_genes,).
    Tiles with zero transcripts get a zero tensor.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_index)}
    n_genes = len(gene_index)
    counts = np.zeros((len(tiles), n_genes), dtype=np.int32)
    tile_id_to_row = {tid: i for i, tid in enumerate(tiles.tile_id)}

    pf = pq.ParquetFile(transcripts_path)
    n_chunks = pf.metadata.num_rows // 1_000_000 + 1
    logger.info(f"Patchifying {pf.metadata.num_rows:,} transcripts across {len(tiles)} tiles ({n_chunks} chunks)")

    for j, batch in enumerate(pf.iter_batches(batch_size=1_000_000, columns=["feature_name", "he_x", "he_y"]), 1):
        logger.info(f"Chunk {j}/{n_chunks}")
        df = batch.to_pandas()
        chunk = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.he_x, df.he_y), crs=tiles.crs)

        joined = gpd.sjoin(chunk, tiles[["tile_id", "geometry"]], how="inner", predicate=predicate)
        if joined.empty:
            continue

        for (tile_id, gene), grp in joined.groupby(["tile_id", "feature_name"]):
            row = tile_id_to_row.get(tile_id)
            col = gene_to_idx.get(gene)
            if row is not None and col is not None:
                counts[row, col] += len(grp)

    logger.info("Writing per-tile transcript tensors")
    for i, tid in enumerate(tiles.tile_id):
        tile_dir = output_dir / str(tid)
        tile_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(counts[i]), tile_dir / "transcripts.pt")

    logger.info("Transcript patchification done")
