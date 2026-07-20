"""Build (or validate) the source/target panel for a filtered item set."""

import sys

from dotenv import load_dotenv
from loguru import logger

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.panel import create_panel
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(artifacts_cfg: ArtifactsConfig, *, overwrite: bool = False) -> None:
    panel_cfg = artifacts_cfg.panel
    assert panel_cfg is not None, 'panel is required'
    assert panel_cfg.name is not None, 'panel.name is required'

    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    panel_path = managed_paths.panels_dir / f'{panel_cfg.name}.yaml'

    if panel_cfg.n_top_genes is None and panel_cfg.flavor is None:
        assert panel_cfg.metadata_path is None, 'panel.metadata_path is only valid for generated panels'
        assert panel_path.exists(), f'Panel not found: {panel_path}'
        logger.info(f'Using predefined panel: {panel_path}')
        return

    assert panel_cfg.n_top_genes is not None, 'panel.n_top_genes is required'
    assert panel_cfg.flavor is not None, 'panel.flavor is required'
    assert panel_cfg.metadata_path is not None, 'panel.metadata_path is required'
    if panel_path.exists() and not overwrite:
        logger.info(f'Panel already exists: {panel_path}')
        return

    items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    split_metadata_path = panel_cfg.metadata_path
    if not split_metadata_path.is_absolute():
        split_metadata_path = managed_paths.output_dir / 'splits' / split_metadata_path
    assert items_path.exists(), f'Items not found: {items_path}'
    assert split_metadata_path.exists(), f'Metadata not found: {split_metadata_path}'
    create_panel(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        processed_dir=managed_paths.processed_dir,
        output_path=panel_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        overwrite=overwrite,
    )


def cli(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.artifacts, overwrite=init.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
