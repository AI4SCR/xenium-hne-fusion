from __future__ import annotations

from pathlib import Path

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
    assert target in TARGETS, f'Unknown target: {target}'
    _assert_columns(runs, ['config.data.name', 'config.data.items_path', 'config.data.metadata_path'])

    matching = runs.apply(lambda row: _matches_artifact_scope(row, artifacts_cfg), axis=1)
    selected = runs.loc[matching].copy()
    logger.info(
        f'Selected {len(selected)}/{len(runs)} W&B runs for '
        f'dataset={artifacts_cfg.name}, items={artifacts_cfg.items.name}, split={artifacts_cfg.split.name}'
    )
    assert not selected.empty, 'No W&B runs match artifact config'

    return selected, _plot_title(artifacts_cfg, target), _output_name(artifacts_cfg, target)


def _matches_artifact_scope(row: pd.Series, artifacts_cfg: ArtifactsConfig) -> bool:
    if row['config.data.name'] != artifacts_cfg.name:
        return False
    if not _matches_path_suffix(row['config.data.items_path'], f'/items/{artifacts_cfg.items.name}.json'):
        return False

    split_stem = split_stem_from_metadata_path(row['config.data.metadata_path'], artifacts_cfg.split.name)
    if split_stem is None:
        return False
    return _matches_panel(row, artifacts_cfg.panel, split_stem=split_stem)


def split_stem_from_metadata_path(value, split_name: str) -> str | None:
    path = _normalize_path(value)
    if path is None:
        return None
    path = f'/{path.lstrip("/")}'

    if path.endswith(f'/splits/{split_name}.parquet'):
        return split_name

    marker = f'/splits/{split_name}/'
    if marker not in path:
        return None

    filename = path.split(marker, maxsplit=1)[1]
    assert '/' not in filename and filename.endswith('.parquet'), f'Expected split parquet: {path}'
    return Path(filename).stem


def _matches_panel(row: pd.Series, panel: PanelConfig | None, *, split_stem: str) -> bool:
    if panel is None:
        return True
    if _is_generated_panel(panel):
        prefix = _generated_panel_prefix(panel)
        return _matches_path_suffix(row.get('config.data.panel_path'), f'/panels/{prefix}{split_stem}.yaml')
    assert panel.name is not None, 'panel.name is required'
    return _matches_path_suffix(row.get('config.data.panel_path'), f'/panels/{panel.name}.yaml')


def _is_generated_panel(panel: PanelConfig) -> bool:
    fields = [panel.metadata_path, panel.n_top_genes, panel.flavor]
    assert all(value is not None for value in fields) or all(value is None for value in fields), 'Invalid panel config'
    return any(value is not None for value in fields)


def _generated_panel_prefix(panel: PanelConfig) -> str:
    assert panel.name is not None, 'panel.name is required'
    assert panel.metadata_path is not None, 'panel.metadata_path is required'
    split_stem = Path(str(panel.metadata_path).replace('\\', '/')).stem
    assert panel.name.endswith(split_stem), f'Panel name must end with split stem: {panel.name}'
    return panel.name.removesuffix(split_stem)


def _matches_path_suffix(value, suffix: str) -> bool:
    path = _normalize_path(value)
    return path is not None and f'/{path.lstrip("/")}'.endswith(suffix)


def _normalize_path(value) -> str | None:
    if _is_missing(value):
        return None
    return str(value).replace('\\', '/')


def _plot_title(artifacts_cfg: ArtifactsConfig, target: str) -> str:
    dataset = DATASET_LABELS.get(artifacts_cfg.name, artifacts_cfg.name)
    artifact = artifacts_cfg.items.name if artifacts_cfg.items.name == artifacts_cfg.split.name else artifacts_cfg.split.name
    return f'{dataset} {artifact} {TARGET_LABELS[target]}'


def _output_name(artifacts_cfg: ArtifactsConfig, target: str) -> str:
    parts = [artifacts_cfg.name, target, artifacts_cfg.items.name]
    if artifacts_cfg.split.name != artifacts_cfg.items.name:
        parts.append(artifacts_cfg.split.name)
    return '-'.join(_clean_name(part) for part in parts)


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
