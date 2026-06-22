"""Process BEAT samples: tissue detection, tiling, transcript and cell extraction."""

import sys

import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.processing import extract_tiles, process_cells, process_tiles, tile_cells, tile_transcripts
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL, build_pipeline_config, select_sample_ids

from pathlib import Path

def normalize_transcripts(transcript_path: Path, output_path: Path):
    import pandas as pd

    logger.info(f'Normalizing transcript file: {transcript_path}')

    table = pd.read_parquet(transcript_path)
    include = table.codeword_category.isin({'predesigned_gene', 'custom_gene'})
    codewords = table.codeword_category.value_counts().to_dict()
    logger.info(f'{codewords}')

    table = table[include]
    geometry = gpd.points_from_xy(x=table.x_location, y=table.y_location)
    table = gpd.GeoDataFrame(table, geometry=geometry)
    table.to_parquet(output_path)


def main(
    data_cfg: DataConfig,
    overwrite: bool = False,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
) -> None:
    assert data_cfg.name in ["owkin"], f"Expected dataset=owkin', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)
    sample_ids = select_sample_ids(
        sorted(p.name for p in cfg.paths.structured_dir.iterdir() if p.is_dir()),
        cfg.data.filter,
    )
    tiles_cfg = cfg.data.tiles
    assert tiles_cfg.img_size is not None, "tiles.img_size is required"
    img_size = tiles_cfg.img_size
    kernel_size = tiles_cfg.kernel_size
    predicate = tiles_cfg.predicate

    for sample_id in sample_ids:
        logger.info(f"Processing BEAT sample {sample_id}")
        structured_dir = cfg.paths.structured_dir / sample_id
        wsi_path = structured_dir / "wsi.tiff"
        transcripts_path = structured_dir / "transcripts.parquet"
        transcripts_norm_path = structured_dir / "transcripts_normalized.parquet"
        cells_path = structured_dir / "cells.parquet"
        tissues_path = structured_dir / "tissues.parquet"
        tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
        processed_dir = cfg.paths.processed_dir / sample_id / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}"

        normalize_transcripts(transcript_path=transcripts_path, output_path=transcripts_norm_path)

        detect_tissues(wsi_path, tissues_path)
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tile_tissues(
            wsi_path,
            tissues_parquet=tissues_path,
            tile_px=tiles_cfg.tile_px,
            stride_px=tiles_cfg.stride_px,
            mpp=tiles_cfg.mpp,
            output_parquet=tiles_path,
        )
        tiles = gpd.read_parquet(tiles_path)
        extract_tiles(wsi_path, tiles, processed_dir, tiles_cfg.mpp, img_size=img_size)
        tile_transcripts(tiles, transcripts_norm_path, processed_dir, img_size=img_size, predicate=predicate)
        process_tiles(tiles, processed_dir, img_size=img_size, kernel_size=kernel_size)
        if cells_path.exists():
            tile_cells(tiles, cells_path, processed_dir, predicate=predicate)
            process_cells(tiles, processed_dir, img_size=img_size, cell_type_col=cell_type_col)


def cli(argv: list[str] | None = None) -> int:
    data_cfg, overwrite, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg, overwrite=overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))