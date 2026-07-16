"""Collect processed tiles into the source items list, and load/join items tables."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from xenium_hne_fusion.artifacts.config import ItemsFilterConfig
from xenium_hne_fusion.metadata import normalize_sample_metadata, read_metadata_table

DEFAULT_SOURCE_ITEMS_NAME = 'all'
STAT_COLS = ['num_transcripts', 'num_unique_transcripts', 'num_cells', 'num_unique_cells']


def compute_item_stats(item: dict, cell_type_col: str) -> dict:
    tile_dir = Path(item['tile_dir'])

    num_transcripts = 0
    num_unique_transcripts = 0
    transcripts_path = tile_dir / 'transcripts.parquet'
    if transcripts_path.exists():
        transcripts = pd.read_parquet(transcripts_path, columns=['feature_name'])
        num_transcripts = len(transcripts)
        num_unique_transcripts = transcripts['feature_name'].nunique()

    num_cells = 0
    num_unique_cells = 0
    cells_path = tile_dir / 'cells.parquet'
    if cells_path.exists():
        cells = pd.read_parquet(cells_path, columns=[cell_type_col])
        num_cells = len(cells)
        num_unique_cells = cells[cell_type_col].nunique()

    return {
        'id': item['id'],
        'num_transcripts': num_transcripts,
        'num_unique_transcripts': num_unique_transcripts,
        'num_cells': num_cells,
        'num_unique_cells': num_unique_cells,
    }


def apply_filter(stats: pd.DataFrame, cfg: ItemsFilterConfig) -> pd.Series:
    mask = pd.Series(True, index=stats.index)
    for field in STAT_COLS:
        threshold = getattr(cfg.filter, field)
        if threshold is None:
            continue
        mask &= stats[field].notna() & (stats[field] >= threshold)
    return mask


def create_items(items_dir: Path, processed_dir: Path, tile_px: int, stride_px: int, overwrite: bool = False) -> Path:
    items_path = items_dir / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    if items_path.exists() and not overwrite:
        logger.info(f"Items already exist: {items_path}")
        return items_path

    tile_paths = sorted(processed_dir.glob(f"*/{tile_px}_{stride_px}/*/tile.pt"))
    items = [
        {
            "id": f"{p.parents[2].name}_{p.parent.name}",
            "sample_id": p.parents[2].name,
            "tile_id": int(p.parent.name),
            "tile_dir": str(p.parent),
        }
        for p in tile_paths
    ]

    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps(items, indent=2))
    logger.info(f"Saved {len(items)} items -> {items_path}")
    return items_path


def load_items_dataframe(items_path: Path) -> pd.DataFrame:

    items_df = pd.read_json(items_path)
    assert not items_df.empty, f"No items found in {items_path}"

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
