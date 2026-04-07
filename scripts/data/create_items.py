"""Build items JSON for TileDataset from processed tile data.

Output: DATA_DIR/03_output/<name>/items/all.json
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from xenium_hne_fusion.utils.getters import DEFAULT_SOURCE_ITEMS_NAME, iter_tile_dirs, load_pipeline_config, tile_item


def main(dataset: str, config_path: Path | None = None, overwrite: bool = False) -> None:
    load_dotenv()
    cfg = load_pipeline_config(dataset, config_path)
    processed_dir = cfg.paths.processed_dir
    items_path = cfg.paths.output_dir / 'items' / f'{DEFAULT_SOURCE_ITEMS_NAME}.json'

    if items_path.exists() and not overwrite:
        logger.info(f'Items already exist: {items_path}')
        _ensure_output_scaffold(cfg.paths.output_dir)
        return

    sample_dirs = sorted([p for p in processed_dir.iterdir() if p.is_dir()])
    logger.info(f'Found {len(sample_dirs)} samples in {processed_dir}')

    items, skipped = [], []
    for sample_dir in tqdm(sample_dirs, desc='Samples'):
        sample_id = sample_dir.name
        for tile_dir in iter_tile_dirs(sample_dir):
            item = tile_item(tile_dir, sample_id, int(tile_dir.name))
            (items if item is not None else skipped).append(item or tile_dir)

    _ensure_output_scaffold(cfg.paths.output_dir)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    with open(items_path, 'w') as f:
        json.dump(items, f, indent=2)

    logger.info(f'Saved {len(items)} items → {items_path}')
    if skipped:
        logger.warning(f'Skipped {len(skipped)} incomplete tile dirs')


def _ensure_output_scaffold(output_dir: Path) -> None:
    items_dir = output_dir / 'items'
    items_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    from jsonargparse import auto_cli
    auto_cli(main)
