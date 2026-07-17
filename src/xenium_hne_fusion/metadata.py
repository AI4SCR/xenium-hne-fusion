
from pathlib import Path

import pandas as pd
from loguru import logger


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
    metadata = normalize_hest1k_metadata(read_metadata_table(metadata_path))

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

    if 'sample_id' not in metadata.columns:
        assert metadata.index.name == 'sample_id', 'Metadata must contain sample_id column'
        metadata = metadata.reset_index()

    assert metadata['sample_id'].notna().all(), 'Metadata contains null sample_id values'
    return metadata


def normalize_hest1k_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    assert 'sample_id' not in metadata.columns, 'HEST metadata should not already contain sample_id'
    assert 'id' in metadata.columns, "HEST metadata must contain 'id'"
    metadata = metadata.rename(columns={'id': 'sample_id'})
    assert metadata['sample_id'].notna().all(), 'HEST metadata contains null sample_id values'
    return metadata
