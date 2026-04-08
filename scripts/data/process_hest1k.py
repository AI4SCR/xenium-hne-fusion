"""Process HEST1k samples using slide MPP from HEST metadata."""

import shutil

import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.download import get_hest_sample_mpp
from xenium_hne_fusion.metadata import get_structured_metadata_path
from xenium_hne_fusion.processing import extract_tiles, process_cells, process_tiles, tile_cells, tile_transcripts
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import build_pipeline_config, resolve_samples


def main(
    processing_cfg: ProcessingConfig,
    overwrite: bool = False,
) -> None:
    assert processing_cfg.name == "hest1k", f"Expected dataset='hest1k', got {processing_cfg.name!r}"
    cfg = build_pipeline_config(processing_cfg)
    metadata_path = get_structured_metadata_path(cfg.paths.structured_dir)
    sample_ids = resolve_samples(cfg, metadata_path)
    tiles_cfg = cfg.processing.tiles
    assert tiles_cfg.img_size is not None, "tiles.img_size is required"
    img_size = tiles_cfg.img_size
    kernel_size = tiles_cfg.kernel_size
    predicate = tiles_cfg.predicate

    for sample_id in sample_ids:
        logger.info(f"Processing HEST1k sample {sample_id}")
        structured_dir = cfg.paths.structured_dir / sample_id
        wsi_path = structured_dir / "wsi.tiff"
        transcripts_path = structured_dir / "transcripts.parquet"
        cells_path = structured_dir / "cells.parquet"
        tissues_path = structured_dir / "tissues.parquet"
        tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
        processed_dir = cfg.paths.processed_dir / sample_id / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}"
        slide_mpp = get_hest_sample_mpp(sample_id, metadata_path)

        if overwrite and processed_dir.exists():
            shutil.rmtree(processed_dir)

        detect_tissues(wsi_path, tissues_path)
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tile_tissues(
            wsi_path,
            tissues_parquet=tissues_path,
            tile_px=tiles_cfg.tile_px,
            stride_px=tiles_cfg.stride_px,
            mpp=tiles_cfg.mpp,
            output_parquet=tiles_path,
            slide_mpp=slide_mpp,
        )
        tiles = gpd.read_parquet(tiles_path)
        extract_tiles(wsi_path, tiles, processed_dir, tiles_cfg.mpp, native_mpp=slide_mpp, img_size=img_size)
        tile_transcripts(tiles, transcripts_path, processed_dir, img_size=img_size, predicate=predicate)
        process_tiles(
            tiles,
            processed_dir,
            img_size=img_size,
            kernel_size=kernel_size,
        )
        if cells_path.exists():
            tile_cells(tiles, cells_path, processed_dir, predicate)
        process_cells(tiles, processed_dir, img_size=img_size)


if __name__ == "__main__":
    processing_cfg, overwrite, _ = parse_processing_args(include_executor=False)
    main(processing_cfg, overwrite=overwrite)
