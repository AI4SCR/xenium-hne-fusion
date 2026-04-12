"""Tile and render BEAT cell annotations across samples with Ray."""

import sys

import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.pipeline import load_ray_module, wait_for_ray_samples
from xenium_hne_fusion.processing import process_cells, tile_cells
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.utils.getters import (
    PipelineConfig,
    build_pipeline_config,
    processed_sample_dir,
    select_sample_ids,
)


def resolve_beat_samples(cfg: PipelineConfig) -> list[str]:
    sample_ids = sorted(path.name for path in cfg.paths.structured_dir.iterdir() if path.is_dir())
    return select_sample_ids(sample_ids, cfg.data.filter)


def process_sample_cells(cfg: PipelineConfig, sample_id: str) -> str:
    tiles_cfg = cfg.data.tiles
    assert tiles_cfg.img_size is not None, "tiles.img_size is required"

    structured_dir = cfg.paths.structured_dir / sample_id
    cells_path = structured_dir / "cells.parquet"
    tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
    output_dir = processed_sample_dir(cfg, sample_id)

    if not cells_path.exists():
        logger.warning(f"Skipping {sample_id}: missing cells.parquet")
        return sample_id

    assert tiles_path.exists(), f"Missing tiles: {tiles_path}"
    assert output_dir.exists(), f"Missing processed sample dir: {output_dir}"

    tiles = gpd.read_parquet(tiles_path)
    tile_cells(tiles, cells_path, output_dir, predicate=tiles_cfg.predicate)
    process_cells(tiles, output_dir, img_size=tiles_cfg.img_size)
    return sample_id


def build_remote_cell_function(ray):
    @ray.remote(num_cpus=8, num_gpus=0)
    def process_sample_cells_remote(cfg: PipelineConfig, sample_id: str) -> str:
        return process_sample_cells(cfg, sample_id)

    return process_sample_cells_remote


def main(data_cfg: DataConfig) -> None:
    assert data_cfg.name == "beat", f"Expected dataset='beat', got {data_cfg.name!r}"

    cfg = build_pipeline_config(data_cfg)
    sample_ids = resolve_beat_samples(cfg)
    logger.info(f"Processing BEAT cells for {len(sample_ids)} samples")

    ray = load_ray_module()
    if not ray.is_initialized():
        ray.init()

    process_sample_cells_remote = build_remote_cell_function(ray)
    futures = [(sample_id, process_sample_cells_remote.remote(cfg, sample_id)) for sample_id in sample_ids]
    wait_for_ray_samples(ray, futures)


def cli(argv: list[str] | None = None) -> int:
    data_cfg, _, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
