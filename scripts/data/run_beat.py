"""Run the end-to-end BEAT pipeline."""

import json
import sys
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.metadata import (
    load_items_dataframe,
    process_dataset_metadata,
)
from xenium_hne_fusion.pipeline import (
    compute_all_tile_stats,
    create_all_items,
    create_split_collection,
    load_ray_module,
    maybe_reset_sample,
    wait_for_ray_samples,
)
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.processing import (
    extract_tiles,
    process_cells,
    process_tiles,
    tile_cells,
    tile_transcripts,
)
from xenium_hne_fusion.structure import structure_metadata, structure_sample
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import (
    DEFAULT_CELL_TYPE_COL,
    DEFAULT_SOURCE_ITEMS_NAME,
    ItemsFilterConfig,
    PipelineConfig,
    apply_filter,
    build_pipeline_config,
    infer_dataset,
    is_sample_processed,
    is_sample_structured,
    mark_sample_processed,
    mark_sample_structured,
    processed_sample_dir,
)


def get_raw_sample_ids(cfg: PipelineConfig) -> list[str]:
    return sorted(path.name for path in cfg.raw_dir.iterdir() if path.is_dir())


def resolve_beat_samples(cfg: PipelineConfig, sample_id: str | None = None) -> list[str]:
    raw_sample_ids = get_raw_sample_ids(cfg)
    if sample_id is not None:
        assert sample_id in raw_sample_ids, f"Unknown sample_id: {sample_id}"
        return [sample_id]
    if cfg.processing.filter.sample_ids is None:
        return raw_sample_ids

    missing = sorted(set(cfg.processing.filter.sample_ids) - set(raw_sample_ids))
    assert not missing, f"Missing raw sample dirs: {missing}"
    return sorted(cfg.processing.filter.sample_ids)


def structure_beat_metadata(cfg: PipelineConfig) -> None:
    metadata_path = cfg.raw_dir / "metadata.parquet"
    if metadata_path.exists():
        structure_metadata(metadata_path, cfg.paths.structured_dir)


def structure_sample_stage(cfg: PipelineConfig, sample_id: str) -> None:
    raw_sample_dir = cfg.raw_dir / sample_id
    wsi_path = raw_sample_dir / "region.tiff"
    transcripts_path = raw_sample_dir / "transcripts" / "transcripts.parquet"
    assert wsi_path.exists(), f"Missing WSI: {wsi_path}"
    assert transcripts_path.exists(), f"Missing transcripts: {transcripts_path}"
    structure_sample(sample_id, wsi_path, transcripts_path, cfg.paths.structured_dir)
    mark_sample_structured(cfg, sample_id)


def detect_sample_tissues(cfg: PipelineConfig, sample_id: str) -> None:
    logger.info(f"Detecting tissues for BEAT sample {sample_id}")
    structured_dir = cfg.paths.structured_dir / sample_id
    wsi_path = structured_dir / "wsi.tiff"
    tissues_path = structured_dir / "tissues.parquet"
    detect_tissues(wsi_path, tissues_path)


def process_sample(
    cfg: PipelineConfig,
    sample_id: str,
    kernel_size: int = 16,
    predicate: str = "within",
    overwrite: bool = False,
) -> None:
    logger.info(f"Processing BEAT sample {sample_id}")
    structured_dir = cfg.paths.structured_dir / sample_id
    tiles_cfg = cfg.processing.tiles
    wsi_path = structured_dir / "wsi.tiff"
    transcripts_path = structured_dir / "transcripts.parquet"
    cells_path = structured_dir / "cells.parquet"
    tissues_path = structured_dir / "tissues.parquet"
    tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
    processed_dir = processed_sample_dir(cfg, sample_id)

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
    extract_tiles(wsi_path, tiles, processed_dir, tiles_cfg.mpp)
    tile_transcripts(tiles, transcripts_path, processed_dir, predicate)
    process_tiles(
        tiles,
        processed_dir,
        transcripts_path,
        img_size=tiles_cfg.tile_px,
        kernel_size=kernel_size,
    )
    if cells_path.exists():
        tile_cells(tiles, cells_path, processed_dir, predicate)
        process_cells(tiles, processed_dir, img_size=tiles_cfg.tile_px)


def create_filtered_items(
    cfg: PipelineConfig,
    source_items_name: str = DEFAULT_SOURCE_ITEMS_NAME,
    overwrite: bool = False,
) -> tuple[Path, int]:
    filter_cfg = ItemsFilterConfig(
        name=cfg.processing.items.name,
        organs=cfg.processing.items.filter.organs,
        num_transcripts=cfg.processing.items.filter.num_transcripts,
        num_unique_transcripts=cfg.processing.items.filter.num_unique_transcripts,
        num_cells=cfg.processing.items.filter.num_cells,
        num_unique_cells=cfg.processing.items.filter.num_unique_cells,
    )
    output_path = cfg.paths.output_dir / "items" / f"{filter_cfg.name}.json"
    if output_path.exists() and not overwrite:
        logger.info(f"Filtered items already exist: {output_path}")
        return output_path, len(load_items_dataframe(output_path))

    items_path = cfg.paths.output_dir / "items" / f"{source_items_name}.json"
    stats_path = cfg.paths.output_dir / "statistics" / f"{source_items_name}.parquet"
    assert items_path.exists(), f"Source items not found: {items_path}"
    assert stats_path.exists(), f"Statistics not found: {stats_path}"

    items_df = load_items_dataframe(items_path)
    stats = pd.read_parquet(stats_path)
    filtered = items_df[items_df["id"].isin(set(stats.index[apply_filter(stats, filter_cfg)]))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered.to_dict("records"), indent=2))
    logger.info(f"Filter {filter_cfg.name}: {len(items_df)} -> {len(filtered)} tiles")
    logger.info(f"Saved filtered items -> {output_path}")
    return output_path, len(filtered)


def _run(
    processing_cfg: ProcessingConfig,
    overwrite: bool,
    executor: Literal["serial", "ray"],
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
) -> None:
    cfg, sample_ids = prepare_driver_context(processing_cfg)
    kernel_size = cfg.processing.tiles.kernel_size
    predicate = cfg.processing.tiles.predicate

    if executor == "serial":
        retained_sample_ids = run_samples_serial(cfg, sample_ids, kernel_size, predicate, overwrite)
    else:
        retained_sample_ids = run_samples_ray(cfg, sample_ids, kernel_size, predicate, overwrite)

    finalize_dataset(
        cfg,
        retained_sample_ids,
        kernel_size,
        cell_type_col,
        overwrite,
    )


def prepare_driver_context(
    config: ProcessingConfig,
) -> tuple[PipelineConfig, list[str]]:
    dataset = infer_dataset(config.name)
    assert dataset == "beat", f"Expected dataset='beat', got {dataset!r}"
    cfg = build_pipeline_config(config)
    structure_beat_metadata(cfg)
    sample_ids = resolve_beat_samples(cfg)
    logger.info(f"Running BEAT pipeline for {len(sample_ids)} samples")
    return cfg, sample_ids


def process_sample_stage(
    cfg: PipelineConfig,
    sample_id: str,
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> None:
    process_sample(
        cfg,
        sample_id,
        kernel_size=kernel_size,
        predicate=predicate,
        overwrite=overwrite,
    )
    mark_sample_processed(cfg, sample_id)


def run_sample_serial(
    cfg: PipelineConfig,
    sample_id: str,
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> str:
    maybe_reset_sample(cfg, sample_id, overwrite)

    if not is_sample_structured(cfg, sample_id):
        structure_sample_stage(cfg, sample_id)

    if is_sample_processed(cfg, sample_id):
        logger.info(f"Skipping {sample_id}: already processed")
        return sample_id

    detect_sample_tissues(cfg, sample_id)
    process_sample_stage(cfg, sample_id, kernel_size, predicate, overwrite)
    return sample_id


def build_remote_sample_functions(ray):
    @ray.remote(num_cpus=1, num_gpus=0)
    def structure_sample_remote(cfg: PipelineConfig, sample_id: str) -> str:
        structure_sample_stage(cfg, sample_id)
        return sample_id

    @ray.remote(num_cpus=4, num_gpus=1)
    def detect_tissues_remote(ref: object, cfg: PipelineConfig, sample_id: str) -> str:
        del ref
        detect_sample_tissues(cfg, sample_id)
        return sample_id

    @ray.remote(num_cpus=8, num_gpus=0)
    def process_sample_remote(
        ref: object,
        cfg: PipelineConfig,
        sample_id: str,
        kernel_size: int,
        predicate: str,
        overwrite: bool,
    ) -> str:
        del ref
        process_sample_stage(cfg, sample_id, kernel_size, predicate, overwrite)
        return sample_id

    return structure_sample_remote, detect_tissues_remote, process_sample_remote


def run_samples_serial(
    cfg: PipelineConfig,
    sample_ids: list[str],
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> list[str]:
    retained_sample_ids = []
    for current_sample_id in sample_ids:
        retained_sample_ids.append(run_sample_serial(cfg, current_sample_id, kernel_size, predicate, overwrite))
    return retained_sample_ids


def run_samples_ray(
    cfg: PipelineConfig,
    sample_ids: list[str],
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> list[str]:
    ray = load_ray_module()
    if not ray.is_initialized():
        ray.init()

    structure_sample_remote, detect_tissues_remote, process_sample_remote = build_remote_sample_functions(ray)

    retained_sample_ids = []
    futures = []
    for current_sample_id in sample_ids:
        maybe_reset_sample(cfg, current_sample_id, overwrite)

        if is_sample_processed(cfg, current_sample_id):
            logger.info(f"Skipping {current_sample_id}: already processed")
            retained_sample_ids.append(current_sample_id)
            continue

        structure_ref = None
        if not is_sample_structured(cfg, current_sample_id):
            structure_ref = structure_sample_remote.remote(cfg, current_sample_id)

        detect_ref = detect_tissues_remote.remote(structure_ref, cfg, current_sample_id)
        process_ref = process_sample_remote.remote(
            detect_ref,
            cfg,
            current_sample_id,
            kernel_size,
            predicate,
            overwrite,
        )
        futures.append((current_sample_id, process_ref))
        retained_sample_ids.append(current_sample_id)

    wait_for_ray_samples(ray, futures)
    return retained_sample_ids


def finalize_dataset(
    cfg: PipelineConfig,
    retained_sample_ids: list[str],
    kernel_size: int,
    cell_type_col: str,
    overwrite: bool,
) -> None:
    process_dataset_metadata(
        dataset="beat",
        metadata_path=cfg.paths.structured_dir / "metadata.parquet",
        output_path=cfg.paths.processed_dir / "metadata.parquet",
        sample_ids=retained_sample_ids,
    )
    create_all_items(cfg, kernel_size=kernel_size, overwrite=overwrite)
    compute_all_tile_stats(cfg, cell_type_col=cell_type_col, overwrite=overwrite)
    filtered_items_path, num_items = create_filtered_items(
        cfg,
        source_items_name=DEFAULT_SOURCE_ITEMS_NAME,
        overwrite=overwrite,
    )
    if num_items == 0:
        logger.warning(f"No items kept for {cfg.processing.items.name}, skipping split creation")
        return
    create_split_collection(
        cfg,
        filtered_items_path,
        overwrite=overwrite,
    )


def main(
    processing_cfg: ProcessingConfig | None = None,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
    executor: Literal["serial", "ray"] = "serial",
) -> None:
    assert processing_cfg is not None, "processing_cfg is required"
    _run(processing_cfg, overwrite=overwrite, executor=executor, cell_type_col=cell_type_col)


def cli(argv: list[str] | None = None) -> None:
    processing_cfg, overwrite, executor = parse_processing_args(argv)
    assert executor is not None
    _run(processing_cfg, overwrite=overwrite, executor=executor)


if __name__ == "__main__":
    cli(sys.argv[1:])
