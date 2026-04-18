"""Filter output/items/all.json using the item thresholds in an artifacts config."""

import sys

from dotenv import load_dotenv

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.pipeline import filter_items
from xenium_hne_fusion.processing_cli import parse_artifacts_args
from xenium_hne_fusion.utils.getters import get_managed_paths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    managed_paths = get_managed_paths(artifacts_cfg.name)
    items_path = managed_paths.output_dir / 'items' / 'all.json'
    output_path = managed_paths.output_dir / 'items' / f'{artifacts_cfg.items.name}.json'
    stats_path = managed_paths.output_dir / 'statistics' / f'{items_path.stem}.parquet'
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
    artifacts_cfg, overwrite_arg = parse_artifacts_args(argv)
    main(artifacts_cfg=artifacts_cfg, overwrite=overwrite_arg)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
