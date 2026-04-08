"""Create or validate a panel from a training config."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

load_dotenv()

from xenium_hne_fusion.hvg import create_panel
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.utils import resolve_training_paths
from xenium_hne_fusion.utils.getters import get_managed_paths


def main(
    config_path: Path,
    overwrite: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    cfg = Config.from_yaml(config_path)
    cfg, _ = resolve_training_paths(cfg)
    panel_path = cfg.data.panel_path
    assert panel_path is not None, 'data.panel_path is required'
    panel_cfg = cfg.panel
    assert panel_cfg.name is not None, 'panel.name is required'
    assert panel_path.name == f'{panel_cfg.name}.yaml', (
        f'panel.name={panel_cfg.name!r} must match data.panel_path={panel_path.name!r}'
    )

    if panel_cfg.n_top_genes is None and panel_cfg.flavor is None:
        assert panel_path.exists(), f'Panel not found: {panel_path}'
        logger.info(f'Using predefined panel: {panel_path}')
        return

    assert panel_cfg.n_top_genes is not None, 'panel.n_top_genes is required'
    assert panel_cfg.flavor is not None, 'panel.flavor is required'
    if panel_path.exists() and not overwrite:
        logger.info(f'Panel already exists: {panel_path}')
        return

    assert cfg.data.name is not None, 'data.name is required'
    assert cfg.data.items_path is not None, 'data.items_path is required'
    assert cfg.data.metadata_path is not None, 'data.metadata_path is required'

    managed_paths = get_managed_paths(cfg.data.name)
    create_panel(
        items_path=cfg.data.items_path,
        split_metadata_path=cfg.data.metadata_path,
        processed_dir=managed_paths.processed_dir,
        output_path=panel_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        batch_size=cfg.data.batch_size if batch_size is None else batch_size,
        num_workers=cfg.data.num_workers if num_workers is None else num_workers,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    namespace = parser.parse_args(sys.argv[1:])
    main(
        config_path=namespace.config,
        overwrite=namespace.overwrite,
        batch_size=namespace.batch_size,
        num_workers=namespace.num_workers,
    )
