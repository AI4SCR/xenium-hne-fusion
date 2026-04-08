"""Filter output/items/all.json using the item thresholds in a processing config."""

from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.pipeline import filter_items_from_items_path
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import ItemsFilterConfig, build_pipeline_config, load_pipeline_config


def main(
    dataset: str,
    config_path: Path | None = None,
    overwrite: bool = False,
    processing_cfg=None,
) -> None:
    load_dotenv()
    cfg = load_pipeline_config(dataset, config_path) if processing_cfg is None else build_pipeline_config(processing_cfg)
    filter_cfg = ItemsFilterConfig(
        name=cfg.processing.items.name,
        organs=cfg.processing.items.filter.organs,
        include_ids=cfg.processing.items.filter.include_ids,
        exclude_ids=cfg.processing.items.filter.exclude_ids,
        num_transcripts=cfg.processing.items.filter.num_transcripts,
        num_unique_transcripts=cfg.processing.items.filter.num_unique_transcripts,
        num_cells=cfg.processing.items.filter.num_cells,
        num_unique_cells=cfg.processing.items.filter.num_unique_cells,
    )

    items_path = cfg.paths.output_dir / 'items' / 'all.json'
    output_path = cfg.paths.output_dir / 'items' / f'{filter_cfg.name}.json'
    metadata_path = cfg.paths.processed_dir / 'metadata.parquet' if filter_cfg.organs is not None else None
    filter_items_from_items_path(
        items_path=items_path,
        output_path=output_path,
        items_filter_cfg=filter_cfg,
        metadata_path=metadata_path,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    import sys

    processing_cfg, overwrite_arg, _ = parse_processing_args(sys.argv[1:], include_executor=False)
    main(
        dataset=processing_cfg.name.split('-', 1)[0],
        config_path=None,
        overwrite=overwrite_arg,
        processing_cfg=processing_cfg,
    )
