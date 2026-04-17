from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from loguru import logger

from xenium_hne_fusion.utils.getters import get_data_dir


DEFAULT_ENTITY = 'chuv'


def default_cache_dir() -> Path:
    return get_data_dir() / '03_output' / 'eval' / 'wandb'


def load_project_runs(
    project: str,
    *,
    entity: str = DEFAULT_ENTITY,
    cache_dir: Path | None = None,
    refresh: bool = False,
    filters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cache_dir = cache_dir or default_cache_dir()
    cache_path = cache_dir / f'{entity}-{project}.parquet'
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    runs = fetch_runs(project, entity=entity, filters=filters or {'state': 'finished'})
    table = runs_to_frame(runs, entity=entity, project=project)

    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path, index=False)
    logger.info(f'Cached {len(table)} W&B runs -> {cache_path}')
    return table


def fetch_runs(
    project: str,
    *,
    entity: str = DEFAULT_ENTITY,
    filters: dict[str, Any] | None = None,
):
    api = wandb.Api()
    runs = list(api.runs(path=f'{entity}/{project}', filters=filters or {'state': 'finished'}))
    logger.info(f'Fetched {len(runs)} W&B runs from {entity}/{project}')
    return runs


def runs_to_frame(runs, *, entity: str, project: str) -> pd.DataFrame:
    rows = [run_to_row(run, entity=entity, project=project) for run in runs]
    assert rows, f'No W&B runs found for {entity}/{project}'
    return pd.DataFrame(rows).convert_dtypes()


def run_to_row(run, *, entity: str, project: str) -> dict[str, Any]:
    summary = getattr(run.summary, '_json_dict', None)
    if summary is None:
        summary = dict(run.summary)

    row = {
        'entity': entity,
        'project': project,
        'run_id': run.id,
        'run_name': run.name,
        'run_created_at': getattr(run, 'created_at', None),
        'run_updated_at': getattr(run, 'updated_at', None),
        'state': run.state,
        'tags': _clean_value(list(getattr(run, 'tags', None) or [])),
    }
    for key, value in _flatten_dict(summary).items():
        row[key] = _clean_value(value)
    for key, value in _flatten_dict(run.config).items():
        row[f'config.{key}'] = _clean_value(value)
    return row


def restrict_to_wandb_filter(
    table: pd.DataFrame,
    project: str,
    *,
    entity: str = DEFAULT_ENTITY,
    filters: dict[str, Any],
) -> pd.DataFrame:
    runs = fetch_runs(project, entity=entity, filters=filters)
    matching_ids = {run.id for run in runs}
    restricted = table.loc[table['run_id'].isin(matching_ids)]
    assert not restricted.empty, f'No cached runs match W&B filters: {filters}'
    logger.info(f'Restricted to {len(restricted)}/{len(table)} cached runs matching W&B filters')
    return restricted


def _flatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    return pd.json_normalize(data).iloc[0].to_dict()


def _clean_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)
