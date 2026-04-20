from __future__ import annotations

import pandas as pd
from loguru import logger

from xenium_hne_fusion.config import ArtifactsConfig, PanelConfig

TARGETS = {'expression', 'cell_types'}
DATASET_LABELS = {'beat': 'BEAT', 'hest1k': 'HEST1K'}
TARGET_LABELS = {'expression': 'expression', 'cell_types': 'cell types'}


def select_artifact_runs(
    runs: pd.DataFrame,
    *,
    artifacts_cfg: ArtifactsConfig,
    target: str,
) -> tuple[pd.DataFrame, str, str]:
    _assert_columns(runs, ['config.data.name', 'config.data.items_path', 'config.data.metadata_path'])
    items_suffix = f'items/{artifacts_cfg.items.name}.json'

    def matches(row: pd.Series) -> bool:
        if row['config.data.name'] != artifacts_cfg.name:
            return False
        if not _path_ends_with(row['config.data.items_path'], items_suffix):
            return False
        if not _metadata_matches_split(row['config.data.metadata_path'], artifacts_cfg.split.name):
            return False
        if artifacts_cfg.panel is not None:
            if not _panel_matches(row.get('config.data.panel_path'), artifacts_cfg.panel, row['config.data.metadata_path']):
                return False
        return True

    selected = runs.loc[runs.apply(matches, axis=1)].copy()
    logger.info(
        f'Selected {len(selected)}/{len(runs)} W&B runs for '
        f'dataset={artifacts_cfg.name}, split={artifacts_cfg.split.name}'
    )
    assert not selected.empty, 'No W&B runs match artifacts config'
    return selected, _plot_title(artifacts_cfg, target), _output_name(artifacts_cfg, target)


def _metadata_matches_split(value, split_name: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    if parent == f'splits/{split_name}' or parent.endswith(f'/splits/{split_name}'):
        return True
    # file-per-split layout: splits/<name>.parquet
    filename = path.rsplit('/', 1)[-1]
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return stem == split_name


def _panel_matches(panel_value, panel_cfg: PanelConfig, metadata_value) -> bool:
    path = _normalize_path(panel_value)
    if path is None:
        return False
    if panel_cfg.metadata_path is None:
        # Static panel: match by name
        return path == f'panels/{panel_cfg.name}.yaml' or path.endswith(f'/panels/{panel_cfg.name}.yaml')
    # HVG per-split panel: match prefix and fold identifier
    fold_stem = panel_cfg.metadata_path.stem  # e.g. 'outer=0-inner=0-seed=0'
    panel_prefix = panel_cfg.name[: -(len(fold_stem) + 1)]  # strip '-<fold>'
    panel_filename = path.rsplit('/', 1)[-1]
    panel_stem = panel_filename.rsplit('.', 1)[0] if '.' in panel_filename else panel_filename
    if not (panel_stem == panel_cfg.name or panel_stem.startswith(panel_prefix + '-')):
        return False
    run_panel_fold = panel_stem[len(panel_prefix) + 1:]
    meta_path = _normalize_path(metadata_value)
    if meta_path is None:
        return False
    meta_filename = meta_path.rsplit('/', 1)[-1]
    run_meta_stem = meta_filename.rsplit('.', 1)[0] if '.' in meta_filename else meta_filename
    return run_panel_fold == run_meta_stem


def _plot_title(artifacts_cfg: ArtifactsConfig, target: str) -> str:
    dataset = DATASET_LABELS.get(artifacts_cfg.name, artifacts_cfg.name.upper())
    items_label = artifacts_cfg.items.name.replace('/', ' ').replace('-', ' ')
    return f'{dataset} {items_label} {TARGET_LABELS.get(target, target)}'


def _output_name(artifacts_cfg: ArtifactsConfig, target: str) -> str:
    parts = [artifacts_cfg.name, target, artifacts_cfg.items.name]
    return '-'.join(_clean_name(p) for p in parts)


def _path_ends_with(value, suffix: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    return path == suffix or path.endswith('/' + suffix)


def _normalize_path(value) -> str | None:
    if _is_missing(value):
        return None
    return str(value).replace('\\', '/')


def _clean_name(value: str) -> str:
    return value.replace('/', '-').replace(' ', '-').lower()


def _assert_columns(runs: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(runs.columns))
    assert not missing, f'Missing W&B columns: {missing}'


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return bool(pd.isna(value))
