import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.eval.experiments import select_artifact_runs
from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.slugs import SlugSpec, validate_slug_specs
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs
from xenium_hne_fusion.processing_cli import build_artifacts_parser, namespace_to_artifacts_config


load_dotenv(override=True)

SLUG_SPECS = {
    'vision': SlugSpec(
        slug='vision',
        label='Vision',
        order=10,
        modality='uni-modal',
        stage=None,
        strategy=None,
        pool=None,
        learnable_gate=None,
        morph_encoder='ViT-S',
        expr_encoder=None,
    ),
    'expr-token': SlugSpec(
        slug='expr-token',
        label='Expression token',
        order=20,
        modality='uni-modal',
        stage=None,
        strategy=None,
        pool='token',
        learnable_gate=None,
        morph_encoder=None,
        expr_encoder='MLP',
    ),
    'expr-tile': SlugSpec(
        slug='expr-tile',
        label='Expression tile',
        order=30,
        modality='uni-modal',
        stage=None,
        strategy=None,
        pool='tile',
        learnable_gate=None,
        morph_encoder=None,
        expr_encoder='MLP',
    ),
    'early-fusion': SlugSpec(
        slug='early-fusion',
        label='Early fusion',
        order=40,
        modality='multi-modal',
        stage='early',
        strategy='add',
        pool='token',
        learnable_gate=False,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
    'early-fusion-gate': SlugSpec(
        slug='early-fusion-gate',
        label='Early fusion gate',
        order=45,
        modality='multi-modal',
        stage='early',
        strategy='add',
        pool='token',
        learnable_gate=True,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
    'late-fusion-token': SlugSpec(
        slug='late-fusion-token',
        label='Late fusion token',
        order=50,
        modality='multi-modal',
        stage='late',
        strategy='add',
        pool='token',
        learnable_gate=False,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
    'late-fusion-token-gate': SlugSpec(
        slug='late-fusion-token-gate',
        label='Late fusion token gate',
        order=55,
        modality='multi-modal',
        stage='late',
        strategy='add',
        pool='token',
        learnable_gate=True,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
    'late-fusion-tile': SlugSpec(
        slug='late-fusion-tile',
        label='Late fusion tile',
        order=60,
        modality='multi-modal',
        stage='late',
        strategy='add',
        pool='tile',
        learnable_gate=False,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
    'late-fusion-tile-gate': SlugSpec(
        slug='late-fusion-tile-gate',
        label='Late fusion tile gate',
        order=65,
        modality='multi-modal',
        stage='late',
        strategy='add',
        pool='tile',
        learnable_gate=True,
        morph_encoder='ViT-S',
        expr_encoder='MLP',
    ),
}
validate_slug_specs(SLUG_SPECS)


def main(
    artifacts_cfg: ArtifactsConfig,
    project: str,
    target: Literal['expression', 'cell_types'],
    refresh: bool = False,
    cache_dir: Path | None = None,
    output_dir: Path = Path('figures/eval'),
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    cache_dir = cache_dir or default_cache_dir()

    table = load_project_runs(project, entity=entity, cache_dir=cache_dir, refresh=refresh)
    runs, title, name = select_artifact_runs(table, artifacts_cfg=artifacts_cfg, target=target)
    plot_metrics(
        runs,
        specs=SLUG_SPECS,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
    )


def _build_parser() -> ArgumentParser:
    parser = build_artifacts_parser(include_overwrite=False)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--target', type=Literal['expression', 'cell_types'], required=True)
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('figures/eval'))
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    return parser


def cli(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    args = namespace.as_dict()
    main(
        artifacts_cfg=namespace_to_artifacts_config(namespace),
        project=args['project'],
        target=args['target'],
        refresh=args['refresh'],
        cache_dir=args['cache_dir'],
        output_dir=args['output_dir'],
        metrics=args['metrics'],
        entity=args['entity'],
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
