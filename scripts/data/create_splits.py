"""Generate tile-level split metadata collection from items, optionally joined with sample metadata."""

from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

load_dotenv()

from xenium_hne_fusion.metadata import (
    build_split_metadata_frame,
    save_split_metadata,
)
from xenium_hne_fusion.processing_cli import build_processing_parser, namespace_to_processing_config
from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.utils.getters import build_pipeline_config


def build_parser() -> ArgumentParser:
    parser = build_processing_parser(include_executor=False)
    parser.add_argument('--with-metadata', type=bool, default=False)
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[ProcessingConfig, bool, bool]:
    ns = build_parser().parse_args(argv)
    return namespace_to_processing_config(ns), ns.overwrite, ns.with_metadata


def main(processing_cfg: ProcessingConfig, overwrite: bool = False, with_metadata: bool = False) -> None:
    cfg = build_pipeline_config(processing_cfg)
    split_cfg = cfg.processing.split

    configured_items_path = cfg.paths.output_dir / 'items' / f'{cfg.processing.items.name}.json'
    source_items_path = cfg.paths.output_dir / 'items' / 'all.json'
    items_path = configured_items_path if configured_items_path.exists() else source_items_path
    metadata_path = cfg.paths.processed_dir / 'metadata.parquet'
    split_dir = cfg.paths.output_dir / 'splits' / split_cfg.name

    if split_dir.exists() and not overwrite:
        logger.info(f'Split metadata already exists: {split_dir}')
        return

    split_metadata = build_split_metadata_frame(
        items_path,
        split_cfg,
        with_metadata=with_metadata,
        sample_metadata_path=metadata_path,
    )
    save_split_metadata(split_metadata, split_dir, split_cfg, overwrite=overwrite)


if __name__ == '__main__':
    import sys

    processing_cfg, overwrite_arg, with_metadata_arg = parse_args(sys.argv[1:])
    main(processing_cfg, overwrite=overwrite_arg, with_metadata=with_metadata_arg)
