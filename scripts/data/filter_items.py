"""Filter output/items/all.json using the item thresholds in an artifacts config."""

from dotenv import load_dotenv

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.pipeline import filter_items_from_items_path
from xenium_hne_fusion.processing_cli import parse_artifacts_args
from xenium_hne_fusion.utils.getters import ItemsFilterConfig, get_managed_paths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    managed_paths = get_managed_paths(artifacts_cfg.name)
    filter_cfg = ItemsFilterConfig(
        name=artifacts_cfg.items.name,
        organs=artifacts_cfg.items.filter.organs,
        include_ids=artifacts_cfg.items.filter.include_ids,
        exclude_ids=artifacts_cfg.items.filter.exclude_ids,
        num_transcripts=artifacts_cfg.items.filter.num_transcripts,
        num_unique_transcripts=artifacts_cfg.items.filter.num_unique_transcripts,
        num_cells=artifacts_cfg.items.filter.num_cells,
        num_unique_cells=artifacts_cfg.items.filter.num_unique_cells,
    )

    items_path = managed_paths.output_dir / 'items' / 'all.json'
    output_path = managed_paths.output_dir / 'items' / f'{filter_cfg.name}.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet' if filter_cfg.organs is not None else None
    filter_items_from_items_path(
        items_path=items_path,
        output_path=output_path,
        items_filter_cfg=filter_cfg,
        metadata_path=metadata_path,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    import sys

    artifacts_cfg, overwrite_arg = parse_artifacts_args(sys.argv[1:])
    main(artifacts_cfg=artifacts_cfg, overwrite=overwrite_arg)
