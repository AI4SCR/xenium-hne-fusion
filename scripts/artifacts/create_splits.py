"""Generate tile-level split metadata collection from items, optionally joined with sample metadata."""

from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

load_dotenv()

from xenium_hne_fusion.metadata import (
    build_split_metadata_frame,
    save_split_metadata,
)
from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.processing_cli import build_artifacts_parser, namespace_to_artifacts_config
from xenium_hne_fusion.utils.getters import get_managed_paths


def build_parser() -> ArgumentParser:
    parser = build_artifacts_parser()
    parser.add_argument('--with-metadata', type=bool, default=False)
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[ArtifactsConfig, bool, bool]:
    ns = build_parser().parse_args(argv)
    return namespace_to_artifacts_config(ns), ns.overwrite, ns.with_metadata


def main(artifacts_cfg: ArtifactsConfig, overwrite: bool = False, with_metadata: bool = False) -> None:
    managed_paths = get_managed_paths(artifacts_cfg.name)
    split_cfg = artifacts_cfg.split
    items_path = managed_paths.output_dir / 'items' / f'{artifacts_cfg.items.name}.json'
    assert items_path.exists(), f'Items not found: {items_path}'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet'
    split_dir = managed_paths.output_dir / 'splits' / split_cfg.name

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

    artifacts_cfg, overwrite_arg, with_metadata_arg = parse_args(sys.argv[1:])
    main(artifacts_cfg, overwrite=overwrite_arg, with_metadata=with_metadata_arg)
