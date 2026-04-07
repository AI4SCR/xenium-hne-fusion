"""Generate tile-level split metadata collection by joining items with sample metadata."""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.metadata import (
    join_items_with_metadata,
    save_split_metadata,
)
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import build_pipeline_config, infer_dataset, load_pipeline_config


def main(
    dataset: str,
    config_path: Path | None = None,
    items_path: Path | None = None,
    metadata_path: Path | None = None,
    overwrite: bool = False,
    processing_cfg=None,
) -> None:
    cfg = load_pipeline_config(dataset, config_path) if processing_cfg is None else build_pipeline_config(processing_cfg)
    split_cfg = cfg.processing.split

    items_path = items_path or (cfg.paths.output_dir / 'items' / 'all.json')
    metadata_path = metadata_path or (cfg.paths.processed_dir / 'metadata.parquet')
    split_dir = cfg.paths.output_dir / 'splits' / split_cfg.split_name

    if split_dir.exists() and not overwrite:
        logger.info(f'Split metadata already exists: {split_dir}')
        return

    joined = join_items_with_metadata(items_path, metadata_path)
    save_split_metadata(joined, split_dir, split_cfg, overwrite=overwrite)


if __name__ == '__main__':
    import sys

    processing_cfg, overwrite_arg, _ = parse_processing_args(sys.argv[1:], include_executor=False)
    main(
        dataset=infer_dataset(processing_cfg.name),
        overwrite=overwrite_arg,
        processing_cfg=processing_cfg,
    )
