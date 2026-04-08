"""Create or validate a panel from a processing config."""

import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.hvg import create_panel
from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.metadata import get_default_split_path
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import build_pipeline_config, get_panels_dir


def main(
    processing_cfg: ProcessingConfig,
    overwrite: bool = False,
    batch_size: int = 256,
    num_workers: int = 10,
) -> None:
    cfg = build_pipeline_config(processing_cfg)
    panel_cfg = cfg.processing.panel
    assert panel_cfg is not None, 'panel config is required'

    output_path = get_panels_dir(cfg.name) / f'{panel_cfg.name}.yaml'
    if panel_cfg.n_top_genes is None and panel_cfg.flavor is None:
        assert output_path.exists(), f'Panel not found: {output_path}'
        logger.info(f'Using predefined panel: {output_path}')
        return

    assert panel_cfg.n_top_genes is not None and panel_cfg.flavor is not None, 'panel config must set both n_top_genes and flavor'
    if output_path.exists() and not overwrite:
        logger.info(f'Panel already exists: {output_path}')
        return

    items_path = cfg.output_dir / 'items' / f'{cfg.processing.items.name}.json'
    split_dir = cfg.output_dir / 'splits' / cfg.processing.split.name
    split_metadata_path = get_default_split_path(split_dir, cfg.processing.split)
    create_panel(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        processed_dir=cfg.processed_dir,
        output_path=output_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    processing_cfg, overwrite, _, _ = parse_processing_args(sys.argv[1:], include_executor=False)
    main(processing_cfg=processing_cfg, overwrite=overwrite)
