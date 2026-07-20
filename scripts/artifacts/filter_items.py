"""Filter output/items/all.json using the item thresholds in an artifacts config."""

import sys

from dotenv import load_dotenv

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.filter import filter_items
from xenium_hne_fusion.artifacts.items import DEFAULT_SOURCE_ITEMS_NAME
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    items_path = managed_paths.items_dir / f'{DEFAULT_SOURCE_ITEMS_NAME}.json'
    output_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    stats_path = managed_paths.statistics_dir / f'{DEFAULT_SOURCE_ITEMS_NAME}.parquet'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet' if artifacts_cfg.items.filter.organs is not None else None
    filter_items(
        items_path=items_path,
        output_path=output_path,
        stats_path=stats_path,
        items_cfg=artifacts_cfg.items,
        metadata_path=metadata_path,
        overwrite=overwrite,
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(artifacts_cfg=init.artifacts, overwrite=init.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
