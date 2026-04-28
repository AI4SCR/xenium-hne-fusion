from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.utils.getters import get_managed_paths

TARGETS = {'expression', 'cell_types'}
DATASET_LABELS = {'beat': 'BEAT', 'hest1k': 'HEST1K'}
TARGET_LABELS = {'expression': 'expression', 'cell_types': 'cell types'}


def select_runs(
    runs: pd.DataFrame,
    *,
    eval_cfg: EvalConfig,
) -> tuple[pd.DataFrame, str, str]:
    filters = eval_cfg.filters
    assert filters.target in TARGETS, f'Unknown target: {filters.target}'
    required_columns = ['config.task.target', 'config.data.name', 'config.data.items_path', 'config.data.metadata_path']
    if filters.panel_paths is not None:
        required_columns.append('config.data.panel_path')
    _assert_columns(runs, required_columns)

    def matches(row: pd.Series) -> bool:
        if row['config.task.target'] != filters.target:
            return False
        if row['config.data.name'] != filters.name:
            return False
        if not _items_path_matches(row['config.data.items_path'], filters.items_path):
            return False
        if not _path_matches_any(row['config.data.metadata_path'], filters.metadata_paths, root='splits'):
            return False
        if filters.panel_paths is None:
            return True
        return _path_matches_any(row['config.data.panel_path'], filters.panel_paths, root='panels')

    selected = runs.loc[runs.apply(matches, axis=1)].copy()
    logger.info(
        f'Selected {len(selected)}/{len(runs)} W&B runs for '
        f'dataset={filters.name}, metadata_paths={filters.metadata_paths}, panel_paths={filters.panel_paths}'
    )
    assert not selected.empty, 'No W&B runs match eval config'
    return selected, _plot_title(eval_cfg), _output_name(eval_cfg)


def _items_path_matches(value, items_filename: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    suffix = f'items/{items_filename}'
    return path == items_filename or path == suffix or path.endswith('/' + suffix)


def _path_matches_any(value, candidates: list[str] | None, *, root: str) -> bool:
    if candidates is None:
        return True
    return any(_path_matches(value, candidate, root=root) for candidate in candidates)


def _path_matches(value, candidate: str, *, root: str) -> bool:
    path = _normalize_path(value)
    if path is None:
        return False
    suffix = f'{root}/{candidate}'
    return path == candidate or path == suffix or path.endswith('/' + suffix)


def _plot_title(eval_cfg: EvalConfig) -> str:
    dataset = DATASET_LABELS.get(eval_cfg.filters.name, eval_cfg.filters.name.upper())
    scope = _scope_label(eval_cfg.filters.metadata_paths)
    return f'{dataset} {scope} {TARGET_LABELS.get(eval_cfg.filters.target, eval_cfg.filters.target)}'


def _output_name(eval_cfg: EvalConfig) -> str:
    parts = [eval_cfg.filters.name, eval_cfg.filters.target]
    return '-'.join(_clean_name(p) for p in parts)


def build_plot_output_prefix(
    runs: pd.DataFrame,
    *,
    eval_cfg: EvalConfig,
    output_dir: Path,
) -> Path:
    return output_dir / _output_name(eval_cfg)


def resolve_eval_output_dir(eval_cfg: EvalConfig, override: Path | None = None) -> Path:
    output_dir = override or eval_cfg.output_dir
    if output_dir.is_absolute():
        return output_dir
    return get_managed_paths(eval_cfg.filters.name).output_dir / output_dir


def _normalize_path(value) -> str | None:
    if _is_missing(value):
        return None
    return str(value).replace('\\', '/')


def _clean_name(value: str) -> str:
    return value.replace('/', '-').replace(' ', '-').lower()


def _assert_columns(runs: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(runs.columns))
    assert not missing, f'Missing W&B columns: {missing}'


def _scope_label(metadata_paths: list[str] | None) -> str:
    if not metadata_paths:
        return 'selected-splits'
    parents = {str(Path(path).parent) for path in metadata_paths}
    if len(parents) == 1:
        return next(iter(parents))
    return 'selected-splits'


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return bool(pd.isna(value))
