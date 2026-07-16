"""Create filtered items, splits, panels, and item stats from an artifacts config."""

import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.items import DEFAULT_SOURCE_ITEMS_NAME, create_items
from xenium_hne_fusion.artifacts.panel import create_panel
from xenium_hne_fusion.artifacts.stats import compute_items_stats, default_stats_paths
from xenium_hne_fusion.artifacts.splits import create_split_collection
from xenium_hne_fusion.artifacts.filter import filter_items
from xenium_hne_fusion.utils.getters import ManagedPaths


def _get_managed_paths(artifacts_cfg: ArtifactsConfig) -> ManagedPaths:
    return ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)


def _filter_items(artifacts_cfg: ArtifactsConfig, *, overwrite: bool) -> None:
    managed_paths = _get_managed_paths(artifacts_cfg)
    items_path = managed_paths.items_dir / f'{DEFAULT_SOURCE_ITEMS_NAME}.json'
    output_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet' if artifacts_cfg.items.filter.organs is not None else None
    filter_items(
        items_path=items_path,
        output_path=output_path,
        stats_path=default_stats_paths(managed_paths, items_path).stats,
        items_cfg=artifacts_cfg.items,
        metadata_path=metadata_path,
        overwrite=overwrite,
    )


def _create_panel(artifacts_cfg: ArtifactsConfig, *, overwrite: bool) -> None:
    panel_cfg = artifacts_cfg.panel
    assert panel_cfg is not None, 'panel is required'

    managed_paths = _get_managed_paths(artifacts_cfg)
    panel_path = managed_paths.panels_dir / f'{panel_cfg.name}.yaml'
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


def main(artifacts_cfg: ArtifactsConfig, overwrite: bool = False) -> None:
    managed_paths = _get_managed_paths(artifacts_cfg)
    source_items_path = create_items(
        managed_paths.items_dir,
        managed_paths.processed_dir,
        tile_px=artifacts_cfg.tile_px,
        stride_px=artifacts_cfg.stride_px,
        overwrite=overwrite,
    )
    compute_items_stats(
        source_items_path,
        managed_paths,
        cell_type_col=artifacts_cfg.cell_type_col,
        overwrite=overwrite,
    )

    filtered_items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
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

    compute_items_stats(
        filtered_items_path,
        managed_paths,
        cell_type_col=artifacts_cfg.cell_type_col,
        overwrite=overwrite,
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(artifacts_cfg=init.artifacts, overwrite=init.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
