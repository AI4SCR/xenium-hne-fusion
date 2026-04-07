"""Run the end-to-end HEST1K human Xenium pipeline."""

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()

from xenium_hne_fusion.config import (
    FilterConfig,
    ItemsConfig,
    ItemsThresholdConfig,
    ProcessingConfig,
    SplitConfig,
    TilesConfig,
)
from xenium_hne_fusion.download import (
    create_structured_metadata_symlink,
    create_structured_symlinks,
    download_hest_metadata,
    download_sample,
    get_hest_sample_mpp,
    validate_hest_sample_mpp,
)
from xenium_hne_fusion.metadata import (
    join_items_with_metadata,
    load_items_dataframe,
    process_dataset_metadata,
    save_split_metadata,
)
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.processing import (
    extract_tiles,
    process_cells,
    process_tiles,
    tile_cells,
    tile_transcripts,
)
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import (
    DEFAULT_CELL_TYPE_COL,
    DEFAULT_SOURCE_ITEMS_NAME,
    STAT_COLS,
    ItemsFilterConfig,
    PipelineConfig,
    apply_filter,
    build_pipeline_config,
    clear_sample_markers,
    compute_item_stats,
    infer_dataset,
    is_sample_processed,
    is_sample_structured,
    iter_tile_dirs,
    load_pipeline_config,
    mark_sample_processed,
    mark_sample_structured,
    processed_sample_dir,
    resolve_samples,
    tile_item,
)


def get_hest_metadata_path(raw_dir: Path) -> Path:
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    if metadata_path.exists():
        return metadata_path
    return download_hest_metadata(raw_dir)


def ensure_hest_sample_downloaded(sample_id: str, raw_dir: Path) -> None:
    wsi_files = list((raw_dir / "wsis").glob(f"{sample_id}*"))
    tx_files = list((raw_dir / "transcripts").glob(f"{sample_id}*"))
    if len(wsi_files) == 1 and len(tx_files) == 1:
        return
    download_sample(sample_id, raw_dir)


def process_sample(
    cfg: PipelineConfig,
    sample_id: str,
    metadata_path: Path,
    kernel_size: int = 16,
    predicate: str = "within",
    overwrite: bool = False,
) -> None:
    logger.info(f"Processing HEST1K sample {sample_id}")
    structured_dir = cfg.paths.structured_dir / sample_id
    tiles_cfg = cfg.processing.tiles
    wsi_path = structured_dir / "wsi.tiff"
    transcripts_path = structured_dir / "transcripts.parquet"
    cells_path = structured_dir / "cells.parquet"
    tissues_path = structured_dir / "tissues.parquet"
    tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
    processed_dir = processed_sample_dir(cfg, sample_id)
    slide_mpp = get_hest_sample_mpp(sample_id, metadata_path)

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
    extract_tiles(wsi_path, tiles, processed_dir, tiles_cfg.mpp, native_mpp=slide_mpp)
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


def can_extract_sample_at_tile_mpp(cfg: PipelineConfig, sample_id: str, metadata_path: Path) -> bool:
    slide_mpp = get_hest_sample_mpp(sample_id, metadata_path)
    if slide_mpp > cfg.processing.tiles.mpp:
        logger.warning(
            f"Skipping {sample_id}: slide_mpp={slide_mpp:.4f} is coarser than tile_mpp={cfg.processing.tiles.mpp:.4f}"
        )
        return False
    return True


def filter_hest_samples_by_tile_mpp(cfg: PipelineConfig, sample_ids: list[str], metadata_path: Path) -> list[str]:
    eligible_sample_ids = []
    for sample_id in sample_ids:
        if can_extract_sample_at_tile_mpp(cfg, sample_id, metadata_path):
            eligible_sample_ids.append(sample_id)
    return eligible_sample_ids


def create_all_items(cfg: PipelineConfig, kernel_size: int = 16, overwrite: bool = False) -> Path:
    items_path = cfg.paths.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    if items_path.exists() and not overwrite:
        logger.info(f"Items already exist: {items_path}")
        return items_path

    sample_dirs = sorted(path for path in cfg.paths.processed_dir.iterdir() if path.is_dir())
    logger.info(f"Building items from {len(sample_dirs)} processed samples")

    items = []
    skipped = []
    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        sample_id = sample_dir.name
        for tile_dir in iter_tile_dirs(sample_dir):
            item = tile_item(tile_dir, sample_id, int(tile_dir.name), kernel_size)
            (items if item is not None else skipped).append(item or tile_dir)

    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps(items, indent=2))
    logger.info(f"Saved {len(items)} items -> {items_path}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} incomplete tile dirs")
    return items_path


def compute_all_tile_stats(
    cfg: PipelineConfig,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> Path:
    stats_path = cfg.paths.output_dir / "statistics" / f"{DEFAULT_SOURCE_ITEMS_NAME}.parquet"
    if stats_path.exists() and not overwrite:
        logger.info(f"Statistics already exist: {stats_path}")
        return stats_path

    items_path = cfg.paths.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    items = load_items_dataframe(items_path).to_dict("records")
    rows = [compute_item_stats(item, cell_type_col) for item in tqdm(items, desc="Tiles")]
    stats = pd.DataFrame(rows).set_index("id")
    assert list(stats.columns) == STAT_COLS, f"Unexpected stats columns: {stats.columns.tolist()}"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f"Saved statistics -> {stats_path}")
    return stats_path


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
    if filter_cfg.organs is not None:
        metadata = pd.read_parquet(cfg.paths.processed_dir / "metadata.parquet")
        allowed_samples = set(metadata.loc[metadata["organ"].isin(filter_cfg.organs), "sample_id"])
        items_df = items_df[items_df["sample_id"].isin(allowed_samples)]

    stats = pd.read_parquet(stats_path)
    filtered = items_df[items_df["id"].isin(set(stats.index[apply_filter(stats, filter_cfg)]))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered.to_dict("records"), indent=2))
    logger.info(f"Filter {filter_cfg.name}: {len(items_df)} -> {len(filtered)} tiles")
    logger.info(f"Saved filtered items -> {output_path}")
    return output_path, len(filtered)


def create_split_collection(
    cfg: PipelineConfig,
    items_path: Path,
    overwrite: bool = False,
) -> Path:
    split_cfg = cfg.processing.split
    split_dir = cfg.paths.output_dir / "splits" / split_cfg.split_name
    if split_dir.exists() and not overwrite:
        logger.info(f"Split metadata already exists: {split_dir}")
        return split_dir

    joined = join_items_with_metadata(items_path, cfg.paths.processed_dir / "metadata.parquet")
    save_split_metadata(joined, split_dir, split_cfg, overwrite=overwrite)
    return split_dir
def load_ray_module():
    return importlib.import_module("ray")


def _run(
    processing_cfg: ProcessingConfig,
    overwrite: bool,
    executor: Literal["serial", "ray"],
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
) -> None:
    cfg, metadata_path, eligible_sample_ids = prepare_driver_context(processing_cfg)
    kernel_size = cfg.processing.tiles.kernel_size
    predicate = cfg.processing.tiles.predicate

    if executor == "serial":
        retained_sample_ids = run_samples_serial(
            cfg,
            eligible_sample_ids,
            metadata_path,
            kernel_size,
            predicate,
            overwrite,
        )
    else:
        retained_sample_ids = run_samples_ray(
            cfg,
            eligible_sample_ids,
            metadata_path,
            kernel_size,
            predicate,
            overwrite,
        )

    finalize_dataset(
        cfg,
        metadata_path,
        retained_sample_ids,
        kernel_size,
        cell_type_col,
        overwrite,
    )


def prepare_driver_context(
    config: ProcessingConfig,
) -> tuple[PipelineConfig, Path, list[str]]:
    dataset = infer_dataset(config.name)
    assert dataset == "hest1k", f"Expected dataset='hest1k', got {dataset!r}"
    cfg = build_pipeline_config(config)

    metadata_path = get_hest_metadata_path(cfg.raw_dir)
    create_structured_metadata_symlink(metadata_path, cfg.paths.structured_dir)
    sample_ids = resolve_samples(cfg, metadata_path)
    eligible_sample_ids = filter_hest_samples_by_tile_mpp(cfg, sample_ids, metadata_path)
    logger.info(f"Running HEST1K pipeline for {len(eligible_sample_ids)} eligible human Xenium samples")
    return cfg, metadata_path, eligible_sample_ids


def maybe_reset_sample(cfg: PipelineConfig, sample_id: str, overwrite: bool) -> None:
    if not overwrite:
        return
    clear_sample_markers(cfg, sample_id)
    processed_dir = processed_sample_dir(cfg, sample_id)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)


def structure_sample(cfg: PipelineConfig, sample_id: str, metadata_path: Path) -> None:
    ensure_hest_sample_downloaded(sample_id, cfg.raw_dir)
    validate_hest_sample_mpp(sample_id, cfg.raw_dir, metadata_path)
    assert can_extract_sample_at_tile_mpp(cfg, sample_id, metadata_path), f"Ineligible sample: {sample_id}"
    create_structured_symlinks(sample_id, cfg.raw_dir, cfg.paths.structured_dir)
    mark_sample_structured(cfg, sample_id)


def detect_sample_tissues(cfg: PipelineConfig, sample_id: str) -> None:
    logger.info(f"Detecting tissues for HEST1K sample {sample_id}")
    structured_dir = cfg.paths.structured_dir / sample_id
    wsi_path = structured_dir / "wsi.tiff"
    tissues_path = structured_dir / "tissues.parquet"
    detect_tissues(wsi_path, tissues_path)


def process_sample_stage(
    cfg: PipelineConfig,
    sample_id: str,
    metadata_path: Path,
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> None:
    process_sample(
        cfg,
        sample_id,
        metadata_path,
        kernel_size=kernel_size,
        predicate=predicate,
        overwrite=overwrite,
    )
    mark_sample_processed(cfg, sample_id)


def run_sample_serial(
    cfg: PipelineConfig,
    sample_id: str,
    metadata_path: Path,
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> str:
    maybe_reset_sample(cfg, sample_id, overwrite)

    if not is_sample_structured(cfg, sample_id):
        structure_sample(cfg, sample_id, metadata_path)

    if is_sample_processed(cfg, sample_id):
        logger.info(f"Skipping {sample_id}: already processed")
        return sample_id

    detect_sample_tissues(cfg, sample_id)
    process_sample_stage(cfg, sample_id, metadata_path, kernel_size, predicate, overwrite)
    return sample_id


def build_remote_sample_functions(ray):
    @ray.remote(num_cpus=1, num_gpus=0)
    def structure_sample_remote(cfg: PipelineConfig, sample_id: str, metadata_path: Path) -> str:
        structure_sample(cfg, sample_id, metadata_path)
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
        metadata_path: Path,
        kernel_size: int,
        predicate: str,
        overwrite: bool,
    ) -> str:
        del ref
        process_sample_stage(cfg, sample_id, metadata_path, kernel_size, predicate, overwrite)
        return sample_id

    return structure_sample_remote, detect_tissues_remote, process_sample_remote


def wait_for_ray_samples(ray, futures: list[tuple[str, object]]) -> None:
    failed_sample_ids = []
    for sample_id, future in futures:
        try:
            ray.get(future)
        except Exception as exc:
            logger.error(f"{sample_id} failed: {exc}")
            failed_sample_ids.append(sample_id)
    if failed_sample_ids:
        raise RuntimeError(f"Failed samples: {failed_sample_ids}")


def run_samples_serial(
    cfg: PipelineConfig,
    sample_ids: list[str],
    metadata_path: Path,
    kernel_size: int,
    predicate: str,
    overwrite: bool,
) -> list[str]:
    retained_sample_ids = []
    for current_sample_id in sample_ids:
        retained_sample_ids.append(
            run_sample_serial(cfg, current_sample_id, metadata_path, kernel_size, predicate, overwrite)
        )
    return retained_sample_ids


def run_samples_ray(
    cfg: PipelineConfig,
    sample_ids: list[str],
    metadata_path: Path,
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
            structure_ref = structure_sample_remote.remote(cfg, current_sample_id, metadata_path)

        detect_ref = detect_tissues_remote.remote(structure_ref, cfg, current_sample_id)
        process_ref = process_sample_remote.remote(
            detect_ref,
            cfg,
            current_sample_id,
            metadata_path,
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
    metadata_path: Path,
    retained_sample_ids: list[str],
    kernel_size: int,
    cell_type_col: str,
    overwrite: bool,
) -> None:
    process_dataset_metadata(
        dataset="hest1k",
        metadata_path=metadata_path,
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
