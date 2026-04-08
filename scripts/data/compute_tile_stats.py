"""Compute per-tile statistics for the items artifact defined by the config."""

from dotenv import load_dotenv

from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.pipeline import compute_tile_stats_from_items, plot_tile_stats
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import build_pipeline_config


def main(
    processing_cfg: ProcessingConfig,
    overwrite: bool = False,
    batch_size: int = 256,
    num_workers: int = 10,
) -> None:
    load_dotenv()
    cfg = build_pipeline_config(processing_cfg)
    items_path = cfg.paths.output_dir / 'items' / f'{cfg.processing.items.name}.json'
    assert items_path.exists(), f'Items not found: {items_path}'
    compute_tile_stats_from_items(
        items_path,
        cfg.paths.output_dir,
        overwrite=overwrite,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == '__main__':
    processing_cfg, overwrite_arg, _, _ = parse_processing_args(include_executor=False)
    main(processing_cfg, overwrite=overwrite_arg)
