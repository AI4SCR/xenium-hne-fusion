import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.eval.experiments import resolve_eval_output_dir, select_runs
from xenium_hne_fusion.eval.tables import save_score_latex_table
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs, restrict_to_wandb_filter


load_dotenv(override=True)

DEFAULT_TABLE_METRICS = ['test/spearman_mean', 'test/pearson_mean', 'test/mse_mean']


def main(
    eval_cfg: EvalConfig,
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
    wandb_filters: dict | None = None,
) -> Path:
    metrics = metrics or DEFAULT_TABLE_METRICS
    cache_dir = cache_dir or default_cache_dir(eval_cfg.filters.name)
    output_dir = resolve_eval_output_dir(eval_cfg, override=output_dir)

    table = load_project_runs(eval_cfg.project, entity=entity, cache_dir=cache_dir, refresh=refresh)
    if wandb_filters:
        table = restrict_to_wandb_filter(table, eval_cfg.project, entity=entity, filters=wandb_filters)
    runs, _, _ = select_runs(table, eval_cfg=eval_cfg)
    output_path = output_dir / f'{eval_cfg.filters.name}-{eval_cfg.filters.target}-score-table.tex'
    return save_score_latex_table(
        runs,
        metrics=metrics,
        output_path=output_path,
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
    return parser


def _eval_config_from_args(args: dict) -> EvalConfig:
    filters = EvalConfig.Filters(**args['filters'])
    return EvalConfig(
        project=args['project'],
        output_dir=args['output_dir'],
        filters=filters,
        baseline=args.get('baseline', 'vision'),
        parameter_columns=args.get('parameter_columns'),
        color_by_splits=args.get('color_by_splits', False),
        sort_by_score=args.get('sort_by_score', True),
    )


def cli(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    args = namespace.as_dict()
    main(
        eval_cfg=_eval_config_from_args(args),
        refresh=args.get('refresh', False),
        cache_dir=args.get('cache_dir'),
        output_dir=args.get('output_dir'),
        metrics=args.get('metrics'),
        entity=args.get('entity', DEFAULT_ENTITY),
        wandb_filters=args.get('wandb_filters'),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
