import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.eval.experiments import get_eval_task, select_experiment_runs
from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.slugs import load_slug_specs
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs


load_dotenv(override=True)


def main(
    dataset: str = 'beat',
    target: str = 'expression',
    organ: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path = Path('figures/eval'),
    slugs_path: Path = Path('configs/eval/slugs.json'),
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    task = get_eval_task(dataset=dataset, target=target)
    specs = load_slug_specs(slugs_path)
    cache_dir = cache_dir or default_cache_dir()

    table = load_project_runs(task.project, entity=entity, cache_dir=cache_dir, refresh=refresh)
    runs, title, name = select_experiment_runs(table, task=task, organ=organ)
    plot_metrics(
        runs,
        specs=specs,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beat')
    parser.add_argument('--target', type=str, default='expression')
    parser.add_argument('--organ', type=str, default=None)
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('figures/eval'))
    parser.add_argument('--slugs-path', type=Path, default=Path('configs/eval/slugs.json'))
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv).as_dict()
    main(**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
