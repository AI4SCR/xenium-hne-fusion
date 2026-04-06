"""Process HEST1k samples using slide MPP from HEST metadata."""

import shutil
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.download import get_hest_sample_mpp
from xenium_hne_fusion.metadata import get_structured_metadata_path
from xenium_hne_fusion.processing import extract_tiles, process_cells, process_tiles, tile_cells, tile_transcripts
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import load_pipeline_config, resolve_samples


def main(
    dataset: str = "hest1k",
    config_path: Path | None = None,
    sample_id: str | None = None,
    kernel_size: int = 16,
    predicate: str = "within",
    overwrite: bool = False,
) -> None:
    assert dataset == "hest1k", f"Expected dataset='hest1k', got {dataset!r}"
    cfg = load_pipeline_config(dataset, config_path)
    metadata_path = get_structured_metadata_path(cfg.structured_dir)
    sample_ids = [sample_id] if sample_id is not None else resolve_samples(cfg, metadata_path)

    for sample_id in sample_ids:
        logger.info(f"Processing HEST1k sample {sample_id}")
        structured_dir = cfg.structured_dir / sample_id
        wsi_path = structured_dir / "wsi.tiff"
        transcripts_path = structured_dir / "transcripts.parquet"
        cells_path = structured_dir / "cells.parquet"
        tissues_path = structured_dir / "tissues.parquet"
        tiles_path = structured_dir / "tiles" / f"{cfg.tile_px}_{cfg.stride_px}.parquet"
        processed_dir = cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}"
        slide_mpp = get_hest_sample_mpp(sample_id, metadata_path)

        if overwrite and processed_dir.exists():
            shutil.rmtree(processed_dir)

        detect_tissues(wsi_path, tissues_path)
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tile_tissues(
            wsi_path,
            tissues_parquet=tissues_path,
            tile_px=cfg.tile_px,
            stride_px=cfg.stride_px,
            mpp=cfg.tile_mpp,
            output_parquet=tiles_path,
            slide_mpp=slide_mpp,
        )
        tiles = gpd.read_parquet(tiles_path)
        extract_tiles(wsi_path, tiles, processed_dir, cfg.tile_mpp, native_mpp=slide_mpp)
        tile_transcripts(tiles, transcripts_path, processed_dir, predicate)
        process_tiles(
            tiles,
            processed_dir,
            transcripts_path,
            img_size=cfg.tile_px,
            kernel_size=kernel_size,
        )
        if cells_path.exists():
            tile_cells(tiles, cells_path, processed_dir, predicate)
            process_cells(tiles, processed_dir, img_size=cfg.tile_px)


if __name__ == "__main__":
    auto_cli(main)
