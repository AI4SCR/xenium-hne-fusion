"""Create or validate a panel from an artifacts config."""

import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.hvg import create_panel
from xenium_hne_fusion.processing_cli import build_artifacts_parser, namespace_to_artifacts_config
from xenium_hne_fusion.utils.getters import get_managed_paths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    panel_cfg = artifacts_cfg.panel
    assert panel_cfg is not None, 'panel is required'

    managed_paths = get_managed_paths(artifacts_cfg.name)
    panel_path = managed_paths.output_dir / 'panels' / f'{panel_cfg.name}.yaml'
    assert panel_cfg.name is not None, 'panel.name is required'

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

    items_path = managed_paths.output_dir / 'items' / f'{artifacts_cfg.items.name}.json'
    split_metadata_path = panel_cfg.metadata_path
    if not split_metadata_path.is_absolute():
        split_metadata_path = managed_paths.output_dir / 'splits' / split_metadata_path
    assert items_path.exists(), f'Items not found: {items_path}'
    assert split_metadata_path.exists(), f'Metadata not found: {split_metadata_path}'

    create_panel_kwargs = {}
    if batch_size is not None:
        create_panel_kwargs['batch_size'] = batch_size
    if num_workers is not None:
        create_panel_kwargs['num_workers'] = num_workers

    create_panel(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        processed_dir=managed_paths.processed_dir,
        output_path=panel_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        overwrite=overwrite,
        **create_panel_kwargs,
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    namespace = parser.parse_args(argv)
    artifacts_cfg = namespace_to_artifacts_config(namespace)
    main(
        artifacts_cfg=artifacts_cfg,
        overwrite=namespace.overwrite,
        batch_size=namespace.batch_size,
        num_workers=namespace.num_workers,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
