from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.utils.getters import get_managed_paths, get_panels_dir

TARGETS = {'expression', 'cell_types'}
DATASET_LABELS = {'beat': 'BEAT', 'hest1k': 'HEST1K'}
TARGET_LABELS = {'expression': 'expression', 'cell_types': 'cell types'}


def resolve_eval_paths(eval_cfg: EvalConfig) -> tuple[Path, Path, Path]:
    name = eval_cfg.data.name
    output_dir = get_managed_paths(name).output_dir
    items_path = output_dir / 'items' / eval_cfg.data.items_path
    panel_path = get_panels_dir(name) / eval_cfg.data.panel_path
    splits_dir = output_dir / 'splits' / eval_cfg.data.split_dir
    return items_path, panel_path, splits_dir


def select_runs(
    runs: pd.DataFrame,
    *,
    eval_cfg: EvalConfig,
) -> tuple[pd.DataFrame, str, str]:
    assert eval_cfg.target in TARGETS, f'Unknown target: {eval_cfg.target}'
    _assert_columns(runs, ['config.data.name', 'config.data.items_path', 'config.data.panel_path', 'config.data.metadata_path'])

    items_path, panel_path, splits_dir = resolve_eval_paths(eval_cfg)

    def matches(row: pd.Series) -> bool:
        if row['config.data.name'] != eval_cfg.data.name:
            return False
        if not _path_equals(row['config.data.items_path'], items_path):
            return False
        if not _path_equals(row['config.data.panel_path'], panel_path):
            return False
        return _path_parent_equals(row['config.data.metadata_path'], splits_dir)

    selected = runs.loc[runs.apply(matches, axis=1)].copy()
    logger.info(
        f'Selected {len(selected)}/{len(runs)} W&B runs for '
        f'dataset={eval_cfg.data.name}, split_dir={eval_cfg.data.split_dir}'
    )
    assert not selected.empty, 'No W&B runs match eval config'
    return selected, _plot_title(eval_cfg), _output_name(eval_cfg)


def _path_equals(value, expected: Path) -> bool:
    path = _normalize_path(value)
    return path is not None and Path(path) == expected


def _path_parent_equals(value, expected_dir: Path) -> bool:
    path = _normalize_path(value)
    return path is not None and Path(path).parent == expected_dir


def _normalize_path(value) -> str | None:
    if _is_missing(value):
        return None
    return str(value).replace('\\', '/')


def _plot_title(eval_cfg: EvalConfig) -> str:
    dataset = DATASET_LABELS.get(eval_cfg.data.name, eval_cfg.data.name)
    return f'{dataset} {eval_cfg.data.split_dir} {TARGET_LABELS[eval_cfg.target]}'


def _output_name(eval_cfg: EvalConfig) -> str:
    parts = [eval_cfg.data.name, eval_cfg.target, eval_cfg.data.split_dir]
    return '-'.join(_clean_name(p) for p in parts)


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
