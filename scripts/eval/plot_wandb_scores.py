import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.eval.experiments import select_runs
from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs, restrict_to_wandb_filter
from xenium_hne_fusion.utils.getters import get_managed_paths


load_dotenv(override=True)


def main(
    eval_cfg: EvalConfig,
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
    wandb_filters: dict | None = None,
    order_by_name: bool = False,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    cache_dir = cache_dir or default_cache_dir(eval_cfg.name)
    output_dir = output_dir or get_managed_paths(eval_cfg.name).output_dir / 'figures' / 'eval'

    table = load_project_runs(eval_cfg.project, entity=entity, cache_dir=cache_dir, refresh=refresh)
    if wandb_filters:
        table = restrict_to_wandb_filter(table, eval_cfg.project, entity=entity, filters=wandb_filters)
    runs, title, name = select_runs(table, eval_cfg=eval_cfg)
    plot_metrics(
        runs,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
        order_by_name=order_by_name,
        parameter_columns=eval_cfg.parameter_columns,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--config', action='config')
    parser.add_class_arguments(EvalConfig, nested_key=None)
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    parser.add_argument('--wandb-filters', type=dict, default=None)
    parser.add_argument('--order-by-name', type=bool, default=False)
    return parser


def cli(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    args = namespace.as_dict()
    eval_cfg = EvalConfig(
        project=args['project'],
        target=args['target'],
        name=args['name'],
        items_path=args['items_path'],
        metadata_dir=args['metadata_dir'],
        parameter_columns=args.get('parameter_columns'),
    )
    main(
        eval_cfg=eval_cfg,
        refresh=args.get('refresh', False),
        cache_dir=args.get('cache_dir'),
        output_dir=args.get('output_dir'),
        metrics=args.get('metrics'),
        entity=args.get('entity', DEFAULT_ENTITY),
        wandb_filters=args.get('wandb_filters'),
        order_by_name=args.get('order_by_name', False),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
