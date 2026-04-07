"""Run the end-to-end HEST1K human Xenium pipeline."""

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger
from tqdm import tqdm

load_dotenv()

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
    load_split_config,
    process_dataset_metadata,
    save_split_metadata,
)
from xenium_hne_fusion.processing import (
    extract_tiles,
    process_cells,
    process_tiles,
    tile_cells,
    tile_transcripts,
)
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import (
    ItemsFilterConfig,
    PipelineConfig,
    load_items_filter_config,
    load_pipeline_config,
    resolve_samples,
)

DEFAULT_ITEMS_CONFIG_DIR = Path("configs/items/hest1k")
DEFAULT_DEFAULT_SPLIT_CONFIG_PATH = Path("configs/splits/hest1k.yaml")
DEFAULT_ORGAN_SPLITS_DIR = Path("configs/splits/hest1k")
DEFAULT_CELL_TYPE_COL = "Level3_grouped"
DEFAULT_SOURCE_ITEMS_NAME = "all"
STAT_COLS = ["num_transcripts", "num_unique_transcripts", "num_cells", "num_unique_cells"]


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


def can_extract_sample_at_tile_mpp(cfg: PipelineConfig, sample_id: str, metadata_path: Path) -> bool:
    slide_mpp = get_hest_sample_mpp(sample_id, metadata_path)
    if slide_mpp > cfg.tile_mpp:
        logger.warning(
            f"Skipping {sample_id}: slide_mpp={slide_mpp:.4f} is coarser than tile_mpp={cfg.tile_mpp:.4f}"
        )
        return False
    return True


def filter_hest_samples_by_tile_mpp(cfg: PipelineConfig, sample_ids: list[str], metadata_path: Path) -> list[str]:
    eligible_sample_ids = []
    for sample_id in sample_ids:
        if can_extract_sample_at_tile_mpp(cfg, sample_id, metadata_path):
            eligible_sample_ids.append(sample_id)
    return eligible_sample_ids


def is_hest_sample_processed(
    cfg: PipelineConfig,
    sample_id: str,
    kernel_size: int,
) -> bool:
    tiles_path = cfg.structured_dir / sample_id / "tiles" / f"{cfg.tile_px}_{cfg.stride_px}.parquet"
    if not tiles_path.exists():
        return False

    try:
        tiles = pd.read_parquet(tiles_path, columns=["tile_id"])
    except Exception as exc:
        logger.warning(f"Failed to read tiles parquet for {sample_id}: {exc}")
        return False

    assert "tile_id" in tiles.columns, f"tile_id missing from {tiles_path}"
    expected_tile_ids = {str(int(tile_id)) for tile_id in tiles["tile_id"].tolist()}
    if not expected_tile_ids:
        logger.warning(f"Tiles parquet is empty for {sample_id}: {tiles_path}")
        return False

    processed_dir = cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}"
    if not processed_dir.exists():
        return False

    observed_tile_ids = {path.name for path in processed_dir.iterdir() if path.is_dir()}
    if observed_tile_ids != expected_tile_ids:
        return False

    required_filenames = {
        "tile.pt",
        "transcripts.parquet",
        f"expr-kernel_size={kernel_size}.parquet",
    }
    for tile_id in expected_tile_ids:
        tile_dir = processed_dir / tile_id
        if not tile_dir.is_dir():
            return False
        if any(not (tile_dir / filename).exists() for filename in required_filenames):
            return False

    return True


def _tile_item(tile_dir: Path, sample_id: str, tile_id: int, kernel_size: int) -> dict | None:
    if not (tile_dir / "tile.pt").exists():
        return None
    if not (tile_dir / f"expr-kernel_size={kernel_size}.parquet").exists():
        return None
    if not (tile_dir / "transcripts.parquet").exists():
        return None
    return {
        "id": f"{sample_id}_{tile_id}",
        "sample_id": sample_id,
        "tile_id": tile_id,
        "tile_dir": str(tile_dir),
    }


def _iter_tile_dirs(sample_dir: Path) -> list[Path]:
    direct_tile_dirs = [path for path in sample_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if direct_tile_dirs:
        return sorted(direct_tile_dirs, key=lambda path: int(path.name))

    config_dirs = [path for path in sample_dir.iterdir() if path.is_dir()]
    assert len(config_dirs) == 1, f"Expected exactly one tile-config dir in {sample_dir}, found {config_dirs}"
    tile_root = config_dirs[0]
    return sorted(
        [path for path in tile_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def create_all_items(cfg: PipelineConfig, kernel_size: int = 16, overwrite: bool = False) -> Path:
    items_path = cfg.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    if items_path.exists() and not overwrite:
        logger.info(f"Items already exist: {items_path}")
        return items_path

    sample_dirs = sorted(path for path in cfg.processed_dir.iterdir() if path.is_dir())
    logger.info(f"Building items from {len(sample_dirs)} processed samples")

    items = []
    skipped = []
    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        sample_id = sample_dir.name
        for tile_dir in _iter_tile_dirs(sample_dir):
            item = _tile_item(tile_dir, sample_id, int(tile_dir.name), kernel_size)
            (items if item is not None else skipped).append(item or tile_dir)

    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps(items, indent=2))
    logger.info(f"Saved {len(items)} items -> {items_path}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} incomplete tile dirs")
    return items_path


def _compute_item_stats(item: dict, cell_type_col: str) -> dict:
    tile_dir = Path(item["tile_dir"])
    transcripts = pd.read_parquet(tile_dir / "transcripts.parquet", columns=["feature_name"])

    num_cells = float("nan")
    num_unique_cells = float("nan")
    cells_path = tile_dir / "cells.parquet"
    if cells_path.exists():
        cells = pd.read_parquet(cells_path, columns=[cell_type_col])
        num_cells = len(cells)
        num_unique_cells = cells[cell_type_col].nunique()

    return {
        "id": item["id"],
        "num_transcripts": len(transcripts),
        "num_unique_transcripts": transcripts["feature_name"].nunique(),
        "num_cells": num_cells,
        "num_unique_cells": num_unique_cells,
    }


def compute_all_tile_stats(
    cfg: PipelineConfig,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> Path:
    stats_path = cfg.output_dir / "statistics" / f"{DEFAULT_SOURCE_ITEMS_NAME}.parquet"
    if stats_path.exists() and not overwrite:
        logger.info(f"Statistics already exist: {stats_path}")
        return stats_path

    items_path = cfg.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    items = load_items_dataframe(items_path).to_dict("records")
    rows = [_compute_item_stats(item, cell_type_col) for item in tqdm(items, desc="Tiles")]
    stats = pd.DataFrame(rows).set_index("id")
    assert list(stats.columns) == STAT_COLS, f"Unexpected stats columns: {stats.columns.tolist()}"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f"Saved statistics -> {stats_path}")
    return stats_path


def _apply_filter(stats: pd.DataFrame, cfg: ItemsFilterConfig) -> pd.Series:
    mask = pd.Series(True, index=stats.index)
    for field in STAT_COLS:
        threshold = getattr(cfg, field)
        if threshold is None:
            continue
        mask &= stats[field].isna() | (stats[field] >= threshold)
    return mask


def create_filtered_items(
    cfg: PipelineConfig,
    items_config_path: Path,
    source_items_name: str = DEFAULT_SOURCE_ITEMS_NAME,
    overwrite: bool = False,
) -> tuple[Path, int]:
    filter_cfg = load_items_filter_config(items_config_path)
    output_path = cfg.output_dir / "items" / f"{filter_cfg.name}.json"
    if output_path.exists() and not overwrite:
        logger.info(f"Filtered items already exist: {output_path}")
        return output_path, len(load_items_dataframe(output_path))

    items_path = cfg.output_dir / "items" / f"{source_items_name}.json"
    stats_path = cfg.output_dir / "statistics" / f"{source_items_name}.parquet"
    assert items_path.exists(), f"Source items not found: {items_path}"
    assert stats_path.exists(), f"Statistics not found: {stats_path}"

    items_df = load_items_dataframe(items_path)
    if filter_cfg.organs is not None:
        metadata = pd.read_parquet(cfg.processed_dir / "metadata.parquet")
        allowed_samples = set(metadata.loc[metadata["organ"].isin(filter_cfg.organs), "sample_id"])
        items_df = items_df[items_df["sample_id"].isin(allowed_samples)]

    stats = pd.read_parquet(stats_path)
    filtered = items_df[items_df["id"].isin(set(stats.index[_apply_filter(stats, filter_cfg)]))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered.to_dict("records"), indent=2))
    logger.info(f"Filter {filter_cfg.name}: {len(items_df)} -> {len(filtered)} tiles")
    logger.info(f"Saved filtered items -> {output_path}")
    return output_path, len(filtered)


def get_split_config_path_for_items(items_config_path: Path) -> Path:
    items_name = load_items_filter_config(items_config_path).name
    path = DEFAULT_DEFAULT_SPLIT_CONFIG_PATH if items_name == "default" else DEFAULT_ORGAN_SPLITS_DIR / f"{items_name}.yaml"
    assert path.exists(), f"Split config not found: {path}"
    return path


def create_split_collection(
    cfg: PipelineConfig,
    split_config_path: Path,
    items_path: Path,
    overwrite: bool = False,
) -> Path:
    split_cfg = load_split_config(split_config_path)
    split_dir = cfg.output_dir / "splits" / split_cfg.split_name
    if split_dir.exists() and not overwrite:
        logger.info(f"Split metadata already exists: {split_dir}")
        return split_dir

    joined = join_items_with_metadata(items_path, cfg.processed_dir / "metadata.parquet")
    save_split_metadata(joined, split_dir, split_cfg, overwrite=overwrite)
    return split_dir


def iter_items_config_paths(items_config_dir: Path) -> list[Path]:
    paths = sorted(items_config_dir.glob("*.yaml"))
    assert paths, f"No item configs found in {items_config_dir}"
    return sorted(paths, key=lambda path: (load_items_filter_config(path).name != "default", path.name))


def select_items_config_paths(
    items_config_dir: Path,
    organ: str | list[str] | None = None,
) -> list[Path]:
    paths = iter_items_config_paths(items_config_dir)
    if organ is None:
        return paths

    organs = [organ] if isinstance(organ, str) else organ
    allowed_names = {"default", *(value.lower() for value in organs)}
    selected = [path for path in paths if load_items_filter_config(path).name.lower() in allowed_names]
    assert selected, f"No matching item configs found for organs={organs} in {items_config_dir}"
    return selected


def main(
    dataset: str = "hest1k",
    config_path: Path | None = None,
    sample_id: str | None = None,
    organ: str | list[str] | None = None,
    items_config_dir: Path = DEFAULT_ITEMS_CONFIG_DIR,
    kernel_size: int = 16,
    predicate: str = "within",
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> None:
    assert dataset == "hest1k", f"Expected dataset='hest1k', got {dataset!r}"
    cfg = load_pipeline_config(dataset, config_path)
    cfg.filter.sample_ids = None
    cfg.filter.species = "Homo sapiens"
    cfg.filter.organ = None
    cfg.filter.disease_type = None

    metadata_path = get_hest_metadata_path(cfg.raw_dir)
    create_structured_metadata_symlink(metadata_path, cfg.structured_dir)
    sample_ids = [sample_id] if sample_id is not None else resolve_samples(cfg, metadata_path)
    eligible_sample_ids = filter_hest_samples_by_tile_mpp(cfg, sample_ids, metadata_path)
    logger.info(f"Running HEST1K pipeline for {len(eligible_sample_ids)} eligible human Xenium samples")

    retained_sample_ids = []
    for current_sample_id in eligible_sample_ids:
        if not overwrite and is_hest_sample_processed(cfg, current_sample_id, kernel_size):
            logger.info(f"Skipping {current_sample_id}: already processed")
            retained_sample_ids.append(current_sample_id)
            continue

        ensure_hest_sample_downloaded(current_sample_id, cfg.raw_dir)
        validate_hest_sample_mpp(current_sample_id, cfg.raw_dir, metadata_path)
        assert can_extract_sample_at_tile_mpp(cfg, current_sample_id, metadata_path), f"Ineligible sample: {current_sample_id}"
        create_structured_symlinks(current_sample_id, cfg.raw_dir, cfg.structured_dir)
        process_sample(
            cfg,
            current_sample_id,
            metadata_path,
            kernel_size=kernel_size,
            predicate=predicate,
            overwrite=overwrite,
        )
        retained_sample_ids.append(current_sample_id)

    process_dataset_metadata(
        dataset="hest1k",
        metadata_path=metadata_path,
        output_path=cfg.processed_dir / "metadata.parquet",
        sample_ids=retained_sample_ids,
    )
    create_all_items(cfg, kernel_size=kernel_size, overwrite=overwrite)
    compute_all_tile_stats(cfg, cell_type_col=cell_type_col, overwrite=overwrite)

    for items_config_path in select_items_config_paths(items_config_dir, organ=organ):
        filtered_items_path, num_items = create_filtered_items(
            cfg,
            items_config_path,
            source_items_name=DEFAULT_SOURCE_ITEMS_NAME,
            overwrite=overwrite,
        )
        if num_items == 0:
            logger.warning(f"No items kept for {items_config_path.stem}, skipping split creation")
            continue
        create_split_collection(
            cfg,
            get_split_config_path_for_items(items_config_path),
            filtered_items_path,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    auto_cli(main)
