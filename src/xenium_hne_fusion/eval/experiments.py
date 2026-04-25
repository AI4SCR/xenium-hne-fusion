from __future__ import annotations

import pandas as pd
from loguru import logger

from xenium_hne_fusion.config import EvalConfig

TARGETS = {'expression', 'cell_types'}
DATASET_LABELS = {'beat': 'BEAT', 'hest1k': 'HEST1K'}
TARGET_LABELS = {'expression': 'expression', 'cell_types': 'cell types'}


def select_runs(
    runs: pd.DataFrame,
    *,
    eval_cfg: EvalConfig,
) -> tuple[pd.DataFrame, str, str]:
    assert eval_cfg.target in TARGETS, f'Unknown target: {eval_cfg.target}'
    _assert_columns(runs, ['config.task.target', 'config.data.name', 'config.data.items_path', 'config.data.metadata_path'])

    def matches(row: pd.Series) -> bool:
        if row['config.task.target'] != eval_cfg.target:
            return False
        if row['config.data.name'] != eval_cfg.name:
            return False
        if not _items_path_matches(row['config.data.items_path'], eval_cfg.items_path):
            return False
        return _metadata_parent_ends_with(row['config.data.metadata_path'], eval_cfg.metadata_dir)

    selected = runs.loc[runs.apply(matches, axis=1)].copy()
    logger.info(
        f'Selected {len(selected)}/{len(runs)} W&B runs for '
        f'dataset={eval_cfg.name}, metadata_dir={eval_cfg.metadata_dir}'
    )
    assert not selected.empty, 'No W&B runs match eval config'
    return selected, _plot_title(eval_cfg), _output_name(eval_cfg)


def _items_path_matches(value, items_filename: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    suffix = f'items/{items_filename}'
    return path == items_filename or path == suffix or path.endswith('/' + suffix)


def _metadata_parent_ends_with(value, metadata_dir: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    suffix = f'splits/{metadata_dir}'
    return parent == suffix or parent.endswith('/' + suffix)


def _plot_title(eval_cfg: EvalConfig) -> str:
    dataset = DATASET_LABELS.get(eval_cfg.name, eval_cfg.name.upper())
    return f'{dataset} {eval_cfg.metadata_dir} {TARGET_LABELS.get(eval_cfg.target, eval_cfg.target)}'


def _output_name(eval_cfg: EvalConfig) -> str:
    parts = [eval_cfg.name, eval_cfg.target, eval_cfg.metadata_dir]
    return '-'.join(_clean_name(p) for p in parts)


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
