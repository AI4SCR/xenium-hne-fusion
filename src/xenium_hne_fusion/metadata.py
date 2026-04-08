
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from ai4bmr_learn.data.splits import Split, save_splits
from loguru import logger

from xenium_hne_fusion.config import SplitConfig


def link_structured_metadata(metadata_path: Path, structured_dir: Path) -> Path:
    metadata_path = metadata_path.resolve()
    assert metadata_path.suffix in {'.csv', '.parquet'}, f'Unsupported metadata format: {metadata_path}'

    dst = structured_dir / f'metadata{metadata_path.suffix}'
    structured_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return dst
    dst.symlink_to(metadata_path)
    return dst


def get_structured_metadata_path(structured_dir: Path) -> Path:
    candidates = [path for path in (structured_dir / 'metadata.csv', structured_dir / 'metadata.parquet') if path.exists()]
    assert len(candidates) == 1, f'Expected exactly one structured metadata file in {structured_dir}, found {candidates}'
    return candidates[0]


def clean_sample_metadata(metadata_path: Path, output_path: Path, selected_sample_ids: list[str] | None = None) -> Path:
    metadata = read_metadata_table(metadata_path)
    metadata = normalize_sample_metadata(metadata)

    if selected_sample_ids is not None:
        keep = metadata['sample_id'].isin(selected_sample_ids)
        metadata = metadata.loc[keep].copy()

    missing = sorted(set(selected_sample_ids or []) - set(metadata['sample_id']))
    assert not missing, f'Some requested sample_ids are missing from metadata: {missing}'
    assert metadata['sample_id'].is_unique, 'Processed metadata must have one row per sample_id'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output_path, index=False)
    logger.info(f'Saved cleaned sample metadata → {output_path}')
    return output_path


def process_hest1k_metadata(
    metadata_path: Path,
    output_path: Path,
    selected_sample_ids: list[str] | None = None,
) -> Path:
    return clean_sample_metadata(metadata_path, output_path, selected_sample_ids=selected_sample_ids)


def process_beat_metadata(
    metadata_path: Path,
    output_path: Path,
    selected_sample_ids: list[str] | None = None,
) -> Path:
    metadata = read_metadata_table(metadata_path)
    if 'sample_id' not in metadata.columns:
        assert metadata.index.name == 'sample_id', 'BEAT metadata must use sample_id index'
        metadata = metadata.reset_index()

    if selected_sample_ids is not None:
        keep = metadata['sample_id'].isin(selected_sample_ids)
        metadata = metadata.loc[keep].copy()

    missing = sorted(set(selected_sample_ids or []) - set(metadata['sample_id']))
    assert not missing, f'Some requested sample_ids are missing from metadata: {missing}'
    assert metadata['sample_id'].is_unique, 'Processed metadata must have one row per sample_id'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output_path, index=False)
    logger.info(f'Saved cleaned sample metadata → {output_path}')
    return output_path


def process_dataset_metadata(
    dataset: str,
    metadata_path: Path,
    output_path: Path,
    selected_sample_ids: list[str] | None = None,
) -> Path:
    match dataset:
        case 'hest1k':
            return process_hest1k_metadata(metadata_path, output_path, selected_sample_ids=selected_sample_ids)
        case 'beat':
            return process_beat_metadata(metadata_path, output_path, selected_sample_ids=selected_sample_ids)
        case _:
            raise ValueError(f'Unsupported dataset for metadata processing: {dataset}')


def read_metadata_table(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.suffix == '.csv':
        return pd.read_csv(metadata_path)
    if metadata_path.suffix == '.parquet':
        return pd.read_parquet(metadata_path)
    raise ValueError(f'Unsupported metadata format: {metadata_path}')


def normalize_sample_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()

    if 'sample_id' in metadata.columns and 'id' in metadata.columns:
        assert metadata['sample_id'].equals(metadata['id']), "Expected 'sample_id' and 'id' to match"
        metadata = metadata.drop(columns=['id'])
    elif 'sample_id' not in metadata.columns:
        assert 'id' in metadata.columns, "Metadata must contain 'sample_id' or 'id'"
        metadata = metadata.rename(columns={'id': 'sample_id'})

    assert metadata['sample_id'].notna().all(), 'Metadata contains null sample_id values'
    return metadata


def load_items_dataframe(items_path: Path) -> pd.DataFrame:
    items = json.loads(items_path.read_text())
    if isinstance(items, dict):
        items = list(items.values())

    items_df = pd.DataFrame(items)
    if items_df.empty:
        return pd.DataFrame(columns=['id', 'sample_id', 'tile_id', 'tile_dir'])

    required = {'id', 'sample_id', 'tile_id', 'tile_dir'}
    missing = required - set(items_df.columns)
    assert not missing, f'Items missing required columns: {sorted(missing)}'
    assert items_df['id'].is_unique, 'Item ids must be unique'
    return items_df


def join_items_with_metadata(items_path: Path, sample_metadata_path: Path) -> pd.DataFrame:
    items_df = load_items_dataframe(items_path)
    metadata_df = normalize_sample_metadata(read_metadata_table(sample_metadata_path))
    assert metadata_df['sample_id'].is_unique, 'Sample metadata must be unique on sample_id'

    metadata_cols = [col for col in metadata_df.columns if col != 'sample_id']
    joined = items_df.merge(metadata_df, on='sample_id', how='left', validate='many_to_one')
    missing = joined[metadata_cols].isna().all(axis=1) if metadata_cols else pd.Series(False, index=joined.index)
    assert not missing.any(), f'Some items are missing sample metadata: {joined.loc[missing, "id"].tolist()}'

    joined = joined.set_index('id', drop=True)
    assert joined.index.is_unique, 'Tile-level metadata index must be unique item ids'
    return joined


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


def load_split_config(path: Path) -> SplitConfig:
    data = yaml.safe_load(path.read_text()) or {}
    return SplitConfig(
        name=data['name'],
        test_size=data.get('test_size'),
        val_size=data.get('val_size'),
        stratify=data.get('stratify', False),
        target_column_name=data.get('target_column_name'),
        encode_targets=data.get('encode_targets', False),
        nan_value=data.get('nan_value', -1),
        use_filtered_targets_for_train=data.get('use_filtered_targets_for_train', False),
        include_targets=data.get('include_targets'),
        group_column_name=data.get('group_column_name'),
        random_state=data.get('random_state'),
    )


def load_named_split_ids(split_root: Path) -> dict[str, list[str]]:
    split_root = Path(split_root)
    split_to_ids: dict[str, list[str]] = {}
    for split in ['train', 'val', 'test']:
        path = split_root / f'{split}.csv'
        assert path.exists(), f'Missing split file: {path}'
        frame = pd.read_csv(path)
        assert 'id' in frame.columns, f"Expected 'id' column in {path}"
        ids = frame['id'].tolist()
        assert len(ids) == len(set(ids)), f'Duplicate sample ids in {path}'
        split_to_ids[split] = ids

    universe = set().union(*map(set, split_to_ids.values()))
    assigned = sum(len(ids) for ids in split_to_ids.values())
    assert assigned == len(universe), f'Overlap across split files in {split_root}'
    return split_to_ids


def save_named_split_metadata(
    joined_metadata: pd.DataFrame,
    split_dir: Path,
    split_to_sample_ids: dict[str, list[str]],
    overwrite: bool = False,
) -> Path:
    if split_dir.exists():
        assert overwrite, f'Split directory already exists: {split_dir}'
        shutil.rmtree(split_dir)

    assert joined_metadata.index.is_unique, 'Tile-level metadata index must be unique'
    joined_metadata = joined_metadata.copy()
    sample_ids = joined_metadata['sample_id']

    split_to_value = {
        'train': Split.FIT.value,
        'val': Split.VAL.value,
        'test': Split.TEST.value,
    }
    assigned_splits = pd.Series(pd.NA, index=joined_metadata.index, dtype='object')

    for source_split, target_split in split_to_value.items():
        split_ids = split_to_sample_ids[source_split]
        mask = sample_ids.isin(split_ids)
        assigned_splits.loc[mask] = target_split

    missing = joined_metadata.loc[assigned_splits.isna(), 'sample_id'].drop_duplicates().tolist()
    assert not missing, f'Samples missing split assignment: {missing}'

    dtype = pd.CategoricalDtype(
        categories=[Split.FIT.value, Split.VAL.value, Split.TEST.value],
        ordered=False,
    )
    joined_metadata[Split.COLUMN_NAME.value] = assigned_splits.astype(dtype)

    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / 'outer=0-inner=0-seed=0.parquet'
    joined_metadata.to_parquet(output_path)
    logger.info(f'Saved explicit split collection → {output_path}')
    return split_dir


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


def get_default_split_path(split_dir: Path, split_cfg: SplitConfig) -> Path:
    seed = split_cfg.random_state
    if split_cfg.val_size is None:
        filename = f'outer=0-seed={seed}.parquet'
    else:
        filename = f'outer=0-inner=0-seed={seed}.parquet'
    path = split_dir / filename
    assert path.exists(), f'Expected canonical split file at {path}'
    return path
