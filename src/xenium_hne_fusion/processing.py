from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from loguru import logger
from wsidata import open_wsi


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


def patchify_transcripts(tiles, transcripts_path: Path, save_dir: Path, predicate: str = "within") -> None:
    # NOTE: this will only create parquet files for tiles that have transcripts!

    transcripts = pq.ParquetFile(transcripts_path)

    logger.info(
        f"Patchify transcripts (num_tiles={len(tiles)}, num_transcripts={transcripts.metadata.num_rows})..."
    )

    chunk_size = 1_000_000
    num_chunks = transcripts.metadata.num_rows // chunk_size + 1
    for j, batch in enumerate(
        transcripts.iter_batches(
            batch_size=chunk_size,
            columns=["transcript_id", "cell_id", "feature_name", "geometry"],
        ),
        start=1,
    ):
        logger.info(f"Processing chunk {j}/{num_chunks}")
        chunk = gpd.GeoDataFrame.from_arrow(batch)

        # NOTE: alternative predicates: 'intersects'.
        # TODO: I am double checking how many transcripts we lose that fall on the border
        joined = gpd.sjoin(chunk, tiles, how="inner", predicate=predicate)
        joined = joined.drop(columns=["index_right"]).to_arrow()

        ds.write_dataset(
            data=joined,
            base_dir=str(save_dir),
            format="parquet",
            basename_template=f"part-{{i}}-chunk={j}.parquet",
            partitioning=["tile_id"],
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
        )

        del chunk, joined
        gc.collect()


def get_patchified_transcripts(tile_id: int, transcripts_dir: Path) -> gpd.GeoDataFrame | None:
    """
    Load transcripts for a single tile from the hive-partitioned dataset.

    Returns None if the tile has no transcripts.
    Reconstructs geometry from WKB so the result is a proper GeoDataFrame.
    """
    tile_dir = transcripts_dir / f"tile_id={tile_id}"
    if not tile_dir.exists():
        return None

    df = pd.read_parquet(tile_dir)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geometry))
