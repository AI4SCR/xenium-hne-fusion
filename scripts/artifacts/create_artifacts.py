"""Create filtered items, splits, panels, and item stats from an artifacts config."""

import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.hvg import create_panel
from xenium_hne_fusion.pipeline import compute_items_stats, create_split_collection, filter_items
from xenium_hne_fusion.processing_cli import parse_artifacts_args
from xenium_hne_fusion.utils.getters import get_managed_paths


def _filter_items(artifacts_cfg: ArtifactsConfig, *, overwrite: bool) -> None:
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


def _create_panel(artifacts_cfg: ArtifactsConfig, *, overwrite: bool) -> None:
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
    create_panel(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        processed_dir=managed_paths.processed_dir,
        output_path=panel_path,
        n_top_genes=panel_cfg.n_top_genes,
        flavor=panel_cfg.flavor,
        overwrite=overwrite,
    )


def main(artifacts_cfg: ArtifactsConfig, overwrite: bool = False) -> None:
    managed_paths = get_managed_paths(artifacts_cfg.name)
    source_items_path = managed_paths.output_dir / 'items' / 'all.json'
    assert source_items_path.exists(), f'Source items not found: {source_items_path}'

    filtered_items_path = managed_paths.output_dir / 'items' / f'{artifacts_cfg.items.name}.json'
    _filter_items(artifacts_cfg=artifacts_cfg, overwrite=overwrite)
    assert filtered_items_path.exists(), f'Filtered items not found: {filtered_items_path}'
    create_split_collection(
        artifacts_cfg.split,
        output_dir=managed_paths.output_dir,
        processed_dir=managed_paths.processed_dir,
        items_path=filtered_items_path,
        overwrite=overwrite,
    )

    if artifacts_cfg.panel is not None:
        _create_panel(artifacts_cfg=artifacts_cfg, overwrite=overwrite)
    else:
        logger.info('Skipping panel creation: no panel config provided')

    compute_items_stats(filtered_items_path, managed_paths.output_dir, overwrite=overwrite)


def cli(argv: list[str] | None = None) -> int:
    artifacts_cfg, overwrite_arg = parse_artifacts_args(argv)
    main(artifacts_cfg=artifacts_cfg, overwrite=overwrite_arg)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
