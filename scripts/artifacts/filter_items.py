"""Filter output/items/all.json using the item thresholds in an artifacts config."""

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
    metadata_path = managed_paths.processed_dir / 'metadata.parquet' if artifacts_cfg.items.filter.organs is not None else None
    filter_items(
        items_path=items_path,
        output_path=output_path,
        items_cfg=artifacts_cfg.items,
        metadata_path=metadata_path,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    import sys

    artifacts_cfg, overwrite_arg = parse_artifacts_args(sys.argv[1:])
    main(artifacts_cfg=artifacts_cfg, overwrite=overwrite_arg)
