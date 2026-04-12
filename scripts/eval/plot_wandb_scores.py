import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.slugs import load_slug_specs
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs


load_dotenv(override=True)

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
    EvalTask(name='beat-expression', project='xe-hne-fus-expr', dataset='beat', target='expression', title='BEAT expression'),
    EvalTask(name='beat-cell-types', project='xe-hne-fus-cell', dataset='beat', target='cell_types', title='BEAT cell types'),
    EvalTask(
        name='hest1k-expression',
        project='xe-hne-fus-expr',
        dataset='hest1k',
        target='expression',
        title='HEST1K expression',
        split_by_organ=True,
    ),
]


def main(
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path = Path('figures/eval'),
    slugs_path: Path = Path('configs/eval/slugs.json'),
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
    datasets: list[str] | None = None,
    targets: list[str] | None = None,
    organs: list[str] | None = None,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    tasks = _select_tasks(datasets=datasets, targets=targets)
    organs = organs or sorted(ORGANS)
    specs = load_slug_specs(slugs_path)
    cache_dir = cache_dir or default_cache_dir()

    tables = {
        project: load_project_runs(project, entity=entity, cache_dir=cache_dir, refresh=refresh)
        for project in sorted({task.project for task in tasks})
    }

    for task in tasks:
        runs = _select_dataset(tables[task.project], task.dataset)
        if task.split_by_organ:
            runs = runs.assign(organ=runs.apply(_infer_organ, axis=1))
            missing = runs.loc[runs['organ'].isna(), 'run_name'].astype(str).tolist()
            assert not missing, f'Could not infer HEST1K organ from W&B tags/config: {missing}'
            runs = runs[runs['organ'].isin(organs)].copy()
            assert not runs.empty, f'No {task.name} runs found for organs: {organs}'
            for organ, organ_runs in runs.groupby('organ', sort=True):
                _plot_task(
                    organ_runs,
                    specs=specs,
                    metrics=metrics,
                    title=f'{task.title}: {organ}',
                    output_dir=output_dir,
                    name=f'{task.name}-{organ}',
                )
        else:
            _plot_task(runs, specs=specs, metrics=metrics, title=task.title, output_dir=output_dir, name=task.name)


def _select_tasks(*, datasets: list[str] | None, targets: list[str] | None) -> list[EvalTask]:
    tasks = TASKS
    unknown = sorted(set(datasets or []) - {'beat', 'hest1k'})
    assert not unknown, f'Unknown datasets: {unknown}'
    unknown = sorted(set(targets or []) - {'expression', 'cell_types'})
    assert not unknown, f'Unknown targets: {unknown}'
    if datasets is not None:
        tasks = [task for task in tasks if task.dataset in datasets]
    if targets is not None:
        tasks = [task for task in tasks if task.target in targets]
    assert tasks, f'No eval tasks selected for datasets={datasets}, targets={targets}'
    return tasks


def _plot_task(
    runs: pd.DataFrame,
    *,
    specs,
    metrics: list[str],
    title: str,
    output_dir: Path,
    name: str,
) -> None:
    logger.info(f'Plotting {title} from {len(runs)} W&B runs')
    outputs = plot_metrics(
        runs,
        specs=specs,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
    )
    for output in outputs:
        logger.info(f'Saved {output}')


def _select_dataset(runs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    assert 'config.data.name' in runs.columns, 'Missing W&B config.data.name'
    selected = runs[runs['config.data.name'] == dataset].copy()
    assert not selected.empty, f'No {dataset} runs found in W&B table'
    return selected


def _infer_organ(row: pd.Series) -> str | None:
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


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('figures/eval'))
    parser.add_argument('--slugs-path', type=Path, default=Path('configs/eval/slugs.json'))
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    parser.add_argument('--datasets', type=list[str], default=None)
    parser.add_argument('--targets', type=list[str], default=None)
    parser.add_argument('--organs', type=list[str], default=None)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv).as_dict()
    main(**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
