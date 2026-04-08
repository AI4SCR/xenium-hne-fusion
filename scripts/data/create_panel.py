"""Create or validate a panel from a processing config."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.hvg import create_panel
from xenium_hne_fusion.metadata import get_default_split_path
from xenium_hne_fusion.utils.getters import get_panels_dir, load_pipeline_config


def main(
    config_path: Path,
    overwrite: bool = False,
) -> None:
    cfg = load_pipeline_config(config_path=config_path)
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
        output_path=output_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    auto_cli(main)
