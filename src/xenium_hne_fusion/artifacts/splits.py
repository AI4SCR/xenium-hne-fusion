"""Build train/val/test split metadata from a filtered items list."""

import shutil
from pathlib import Path

import pandas as pd
from ai4bmr_learn.data.splits import save_splits
from loguru import logger

from xenium_hne_fusion.artifacts.config import SplitConfig
from xenium_hne_fusion.artifacts.items import join_items_with_metadata, load_items_dataframe


def build_split_metadata_frame(
    items_path: Path,
    split_cfg: SplitConfig,
    *,
    with_metadata: bool = False,
    sample_metadata_path: Path | None = None,
) -> pd.DataFrame:
    if with_metadata:
        assert sample_metadata_path is not None, 'sample_metadata_path is required when with_metadata=true'
        return join_items_with_metadata(items_path, sample_metadata_path)

    items_df = load_items_dataframe(items_path).set_index('id', drop=True)
    assert items_df.index.is_unique, 'Tile-level metadata index must be unique item ids'

    required_columns = [
        split_cfg.group_column_name,
        split_cfg.target_column_name,
    ]
    missing = sorted({col for col in required_columns if col is not None and col not in items_df.columns})
    assert not missing, f'Missing split columns: {missing}'
    if split_cfg.include_targets is not None:
        assert split_cfg.target_column_name is not None, 'include_targets requires target_column_name'
        assert split_cfg.target_column_name in items_df.columns, f'Missing split columns: {[split_cfg.target_column_name]}'

    return items_df


def save_split_metadata(
    joined_metadata: pd.DataFrame,
    split_dir: Path,
    split_cfg: SplitConfig,
    overwrite: bool = False,
) -> Path:
    if split_dir.exists():
        assert overwrite, f'Split directory already exists: {split_dir}'
        shutil.rmtree(split_dir)

    save_splits(
        metadata=joined_metadata,
        save_dir=split_dir,
        test_size=split_cfg.test_size or 0.2,
        val_size=split_cfg.val_size,
        stratify=split_cfg.stratify,
        target_column_name=split_cfg.target_column_name,
        encode_targets=split_cfg.encode_targets,
        nan_value=split_cfg.nan_value,
        use_filtered_targets_for_train=split_cfg.use_filtered_targets_for_train,
        include_targets=split_cfg.include_targets,
        group_column_name=split_cfg.group_column_name,
        random_state=split_cfg.random_state,
        overwrite=overwrite,
    )
    logger.info(f'Saved split collection → {split_dir}')
    return split_dir


def create_split_collection(
    split_cfg: SplitConfig,
    *,
    output_dir: Path,
    processed_dir: Path,
    items_path: Path,
    overwrite: bool = False,
) -> Path:
    split_dir = output_dir / "splits" / split_cfg.name
    if split_dir.exists() and not overwrite:
        logger.info(f"Split metadata already exists: {split_dir}")
        return split_dir

    split_metadata = build_split_metadata_frame(
        items_path,
        split_cfg,
        with_metadata=False,
        sample_metadata_path=processed_dir / "metadata.parquet",
    )
    save_split_metadata(split_metadata, split_dir, split_cfg, overwrite=overwrite)
    return split_dir
