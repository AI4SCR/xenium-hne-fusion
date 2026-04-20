import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import ArtifactsConfig, ItemsConfig, PanelConfig, SplitConfig
from xenium_hne_fusion.eval.experiments import select_artifact_runs
from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs, restrict_to_wandb_filter
from xenium_hne_fusion.utils.getters import get_managed_paths


load_dotenv(override=True)


def main(
    artifacts_cfg: ArtifactsConfig,
    project: str,
    target: str,
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
    wandb_filters: dict | None = None,
    order_by_name: bool = False,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    cache_dir = cache_dir or default_cache_dir(artifacts_cfg.name)
    output_dir = output_dir or get_managed_paths(artifacts_cfg.name).output_dir / 'figures' / 'eval'

    table = load_project_runs(project, entity=entity, cache_dir=cache_dir, refresh=refresh)
    if wandb_filters:
        table = restrict_to_wandb_filter(table, project, entity=entity, filters=wandb_filters)
    runs, title, name = select_artifact_runs(table, artifacts_cfg=artifacts_cfg, target=target)
    plot_metrics(
        runs,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
        order_by_name=order_by_name,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--config', action='config')
    parser.add_class_arguments(ArtifactsConfig, nested_key=None)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    parser.add_argument('--wandb-filters', type=dict, default=None)
    parser.add_argument('--order-by-name', type=bool, default=False)
    return parser


def _build_artifacts_cfg(args: dict) -> ArtifactsConfig:
    items_raw = args.get('items') or {}
    split_raw = args.get('split') or {}
    panel_raw = args.get('panel')

    items_cfg = ItemsConfig(**items_raw) if isinstance(items_raw, dict) else items_raw
    split_cfg = SplitConfig(**split_raw) if isinstance(split_raw, dict) else split_raw

    if panel_raw is None:
        panel_cfg = None
    elif isinstance(panel_raw, dict):
        mp = panel_raw.get('metadata_path')
        panel_cfg = PanelConfig(
            name=panel_raw.get('name'),
            metadata_path=Path(mp) if mp is not None else None,
            n_top_genes=panel_raw.get('n_top_genes'),
            flavor=panel_raw.get('flavor'),
        )
    else:
        panel_cfg = panel_raw

    return ArtifactsConfig(
        name=args['name'],
        items=items_cfg,
        split=split_cfg,
        panel=panel_cfg,
    )


def cli(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    args = namespace.as_dict()
    artifacts_cfg = _build_artifacts_cfg(args)
    main(
        artifacts_cfg=artifacts_cfg,
        project=args['project'],
        target=args['target'],
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
