"""Generate tile-level split metadata collection by joining items with sample metadata."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.metadata import (
    join_items_with_metadata,
    load_split_config,
    save_split_metadata,
)
from xenium_hne_fusion.utils.getters import load_pipeline_config


def main(
    dataset: str,
    config_path: Path | None = None,
    split_config_path: Path | None = None,
    items_path: Path | None = None,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    cfg = load_pipeline_config(dataset, config_path)
    split_config_path = split_config_path or Path('configs/splits') / f'{dataset}.yaml'
    split_cfg = load_split_config(split_config_path)

    items_path = items_path or (cfg.paths.output_dir / 'items' / 'all.json')
    metadata_path = metadata_path or (cfg.paths.processed_dir / 'metadata.parquet')
    split_dir = cfg.paths.output_dir / 'splits' / split_cfg.split_name

    if split_dir.exists() and not overwrite:
        logger.info(f'Split metadata already exists: {split_dir}')
        return

    joined = join_items_with_metadata(items_path, metadata_path)
    save_split_metadata(joined, split_dir, split_cfg, overwrite=overwrite)


if __name__ == '__main__':
    auto_cli(main)
