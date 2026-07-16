"""Generate tile-level split metadata collection from items, optionally joined with sample metadata."""

import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.artifacts.splits import (
    build_split_metadata_frame,
    save_split_metadata,
)
from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(artifacts_cfg: ArtifactsConfig, overwrite: bool = False, with_metadata: bool = False) -> None:
    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    split_cfg = artifacts_cfg.split
    items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
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


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    parser.add_argument('--with-metadata', type=bool, default=False)

    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.artifacts, overwrite=init.overwrite, with_metadata=init.with_metadata)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
