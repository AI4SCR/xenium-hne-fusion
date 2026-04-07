"""Cache fixed HESCAPE sample-level splits and materialize them as tile-level parquet splits."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.metadata import join_items_with_metadata, load_named_split_ids, save_named_split_metadata
from xenium_hne_fusion.utils.getters import get_repo_root, load_pipeline_config


@dataclass(frozen=True)
class HescapeSplitRecipe:
    source_dir: str
    items_name: str
    split_name: str


HESCAPE_SPLIT_RECIPES = {
    'breast': HescapeSplitRecipe(
        source_dir='human-breast-panel',
        items_name='breast',
        split_name='hescape-breast',
    ),
    'lung': HescapeSplitRecipe(
        source_dir='human-lung-healthy-panel',
        items_name='lung',
        split_name='hescape-lung-healthy',
    ),
    'colon': HescapeSplitRecipe(
        source_dir='human-colon-panel',
        items_name='colon',
        split_name='hescape-colon',
    ),
}


def main(
    recipe: str,
    dataset: str = 'hest1k',
    config_path: Path | None = None,
    items_path: Path | None = None,
    split_name: str | None = None,
    source_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    assert recipe in HESCAPE_SPLIT_RECIPES, f'Unknown recipe: {recipe}'
    cfg = load_pipeline_config(dataset, config_path)
    spec = HESCAPE_SPLIT_RECIPES[recipe]

    source_dir = source_dir or (get_repo_root() / 'splits' / dataset / 'hescape' / spec.source_dir)
    items_path = items_path or (cfg.paths.output_dir / 'items' / f'{spec.items_name}.json')
    split_name = split_name or spec.split_name

    assert items_path.exists(), f'Items not found: {items_path}'
    split_ids = load_named_split_ids(source_dir)
    source_sample_ids = set().union(*map(set, split_ids.values()))
    logger.info(f'Loaded HESCAPE split from {source_dir} with {len(source_sample_ids)} sample ids')

    joined = join_items_with_metadata(items_path, cfg.paths.processed_dir / 'metadata.parquet')
    local_sample_ids = set(joined['sample_id'])
    missing = sorted(local_sample_ids - source_sample_ids)
    assert not missing, f'Items include samples outside HESCAPE split: {missing}'

    unavailable = sorted(source_sample_ids - local_sample_ids)
    if unavailable:
        logger.info(f'HESCAPE samples unavailable locally: {unavailable}')

    split_dir = cfg.paths.output_dir / 'splits' / split_name
    save_named_split_metadata(joined, split_dir, split_ids, overwrite=overwrite)
    return split_dir


if __name__ == '__main__':
    auto_cli(main)
