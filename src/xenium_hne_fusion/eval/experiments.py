from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


ORGANS = {'bowel', 'breast', 'lung', 'pancreas'}


@dataclass(frozen=True)
class EvalTask:
    name: str
    project: str
    dataset: Literal['beat', 'hest1k']
    target: Literal['expression', 'cell_types']
    title: str
    split_by_organ: bool = False


TASKS = [
    EvalTask(
        name='beat-expression',
        project='xe-hne-fus-expr',
        dataset='beat',
        target='expression',
        title='BEAT expression',
    ),
    EvalTask(
        name='beat-cell-types',
        project='xe-hne-fus-cell',
        dataset='beat',
        target='cell_types',
        title='BEAT cell types',
    ),
    EvalTask(
        name='hest1k-expression',
        project='xe-hne-fus-expr',
        dataset='hest1k',
        target='expression',
        title='HEST1K expression',
        split_by_organ=True,
    ),
]


def get_eval_task(*, dataset: str, target: str) -> EvalTask:
    matches = [task for task in TASKS if task.dataset == dataset and task.target == target]
    assert len(matches) == 1, f'Unknown eval task: dataset={dataset!r}, target={target!r}'
    return matches[0]


def select_experiment_runs(
    runs: pd.DataFrame,
    *,
    task: EvalTask,
    organ: str | None,
) -> tuple[pd.DataFrame, str, str]:
    runs = select_dataset_runs(runs, task.dataset)
    title = task.title
    name = task.name
    if not task.split_by_organ:
        assert organ is None, f'Organ is only valid for organ-split tasks, got {organ!r}'
        return runs, title, name

    assert organ in ORGANS, f'Valid organ is required for {task.name}: {sorted(ORGANS)}'
    runs = select_organ_runs(runs, organ=organ)
    return runs, f'{title}: {organ}', f'{name}-{organ}'


def select_dataset_runs(runs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    assert 'config.data.name' in runs.columns, 'Missing W&B config.data.name'
    selected = runs[runs['config.data.name'] == dataset].copy()
    assert not selected.empty, f'No {dataset} runs found in W&B table'
    return selected


def select_organ_runs(runs: pd.DataFrame, *, organ: str) -> pd.DataFrame:
    runs = runs.assign(organ=runs.apply(infer_organ, axis=1))
    missing = runs.loc[runs['organ'].isna(), 'run_name'].astype(str).tolist()
    assert not missing, f'Could not infer HEST1K organ from W&B tags/config: {missing}'
    selected = runs[runs['organ'] == organ].copy()
    assert not selected.empty, f'No HEST1K expression runs found for organ: {organ}'
    return selected


def infer_organ(row: pd.Series) -> str | None:
    for key in ['config.wandb.tags', 'tags']:
        organ = _organ_from_tags(row.get(key))
        if organ is not None:
            return organ

    for key in ['config.data.items_path', 'config.data.metadata_path', 'config.data.panel_path']:
        value = row.get(key)
        if _is_missing(value):
            continue
        for part in str(value).replace('\\', '/').replace('-', '/').split('/'):
            if part in ORGANS:
                return part
    return None


def _organ_from_tags(value) -> str | None:
    tags = _coerce_tags(value)
    organs = sorted(set(tags) & ORGANS)
    assert len(organs) <= 1, f'Multiple organ tags found: {organs}'
    return organs[0] if organs else None


def _coerce_tags(value) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        assert isinstance(parsed, list), f'Expected list tags, got {type(parsed)}'
        return [str(tag) for tag in parsed]
    if isinstance(value, (list, tuple, set)):
        return [str(tag) for tag in value]
    return [str(value)]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return bool(pd.isna(value))
