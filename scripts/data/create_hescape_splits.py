"""Materialize fixed HESCAPE sample splits from HEST1K all.json."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import join_items_with_metadata, load_named_split_ids
from xenium_hne_fusion.utils.getters import get_managed_paths


def create_hescape_split(
    name: str,
    splits_dir: Path,
    overwrite: bool = False,
) -> Path:
    managed_paths = get_managed_paths('hest1k')
    items_path = managed_paths.output_dir / 'items' / 'all.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet'
    output_path = managed_paths.output_dir / 'splits' / f'{name}.parquet'

    assert items_path.exists(), f'Items not found: {items_path}'
    assert metadata_path.exists(), f'Metadata not found: {metadata_path}'
    if output_path.exists():
        assert overwrite, f'Split already exists: {output_path}'

    split_to_ids = load_named_split_ids(splits_dir)
    split_ids = set().union(*map(set, split_to_ids.values()))

    joined = join_items_with_metadata(items_path, metadata_path)
    keep = joined['sample_id'].isin(split_ids)
    filtered = joined.loc[keep].copy()

    present = set(filtered['sample_id'])
    missing = sorted(split_ids - present)
    assert not missing, f'Split samples missing from items: {missing}'

    split_to_value = {
        'train': 'fit',
        'val': 'val',
        'test': 'test',
    }
    filtered['split'] = None
    for source_split, target_split in split_to_value.items():
        mask = filtered['sample_id'].isin(split_to_ids[source_split])
        filtered.loc[mask, 'split'] = target_split

    assert filtered.index.is_unique, 'Tile-level metadata index must be unique'
    assert filtered['split'].notna().all(), 'Some tiles are missing split labels'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path)
    return output_path


if __name__ == '__main__':
    auto_cli(create_hescape_split)
