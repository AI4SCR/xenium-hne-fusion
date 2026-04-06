"""Filter items/all.json by per-tile statistics thresholds.

Reads statistics/all.parquet (produced by compute_tile_stats.py) and a
per-dataset items config YAML, then writes a filtered items JSON.

Output: DATA_DIR/03_output/<name>/items/<config.name>.json
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

from xenium_hne_fusion.metadata import load_items_dataframe
from xenium_hne_fusion.utils.getters import ItemsFilterConfig, load_items_filter_config, load_pipeline_config

_FILTER_FIELDS = ['num_transcripts', 'num_unique_transcripts', 'num_cells', 'num_unique_cells']


def _apply_filter(stats: pd.DataFrame, cfg: ItemsFilterConfig) -> pd.Series:
    mask = pd.Series(True, index=stats.index)
    for field in _FILTER_FIELDS:
        threshold = getattr(cfg, field)
        if threshold is None:
            continue
        col = stats[field]
        # NaN values (e.g. missing cells data) pass the filter
        mask &= col.isna() | (col >= threshold)
    return mask


def main(
    dataset: str,
    items_config_path: Path,
    source_items_name: str = 'all',
    config_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    cfg = load_pipeline_config(dataset, config_path)
    filter_cfg = load_items_filter_config(items_config_path)

    output_path = cfg.output_dir / 'items' / f'{filter_cfg.name}.json'
    if output_path.exists() and not overwrite:
        logger.info(f'Filtered items already exist: {output_path}')
        return

    items_path = cfg.output_dir / 'items' / f'{source_items_name}.json'
    stats_path = cfg.output_dir / 'statistics' / f'{source_items_name}.parquet'

    assert items_path.exists(), f'Source items not found: {items_path}'
    assert stats_path.exists(), (
        f'Statistics not found: {stats_path}. Run compute_tile_stats.py first.'
    )

    items_df = load_items_dataframe(items_path)

    if filter_cfg.organs is not None:
        meta = pd.read_parquet(cfg.processed_dir / 'metadata.parquet')
        allowed_samples = set(meta.loc[meta.organ.isin(filter_cfg.organs), 'sample_id'])
        items_df = items_df[items_df['sample_id'].isin(allowed_samples)]

    stats = pd.read_parquet(stats_path)

    mask = _apply_filter(stats, filter_cfg)
    kept_ids = set(stats.index[mask])
    filtered = items_df[items_df['id'].isin(kept_ids)]

    n_total = len(items_df)
    n_kept = len(filtered)
    logger.info(f'Filter: {n_total} → {n_kept} tiles ({100 * n_kept / n_total:.1f}%)')

    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered.to_dict('records'), f, indent=2)
    logger.info(f'Saved filtered items → {output_path}')


if __name__ == '__main__':
    auto_cli(main)
