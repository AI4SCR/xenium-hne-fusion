import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import EvalConfig, EvalDataConfig
from xenium_hne_fusion.eval.experiments import select_runs
from xenium_hne_fusion.eval.plotting import METRIC_LABELS, plot_metrics
from xenium_hne_fusion.eval.slugs import SlugSpec, validate_slug_specs
from xenium_hne_fusion.eval.wandb import DEFAULT_ENTITY, default_cache_dir, load_project_runs, restrict_to_wandb_filter
from xenium_hne_fusion.utils.getters import get_managed_paths


load_dotenv(override=True)

def _spec(slug: str, label: str, order: int, modality: str, **kwargs) -> SlugSpec:
    return SlugSpec(slug=slug, label=label, order=order, modality=modality, **kwargs)


SLUG_SPECS = {
    # ── uni-modal ──────────────────────────────────────────────────────────────
    'vision': _spec('vision', 'Vision', 10, 'uni-modal',
        stage=None, strategy=None, pool=None, learnable_gate=None,
        morph_encoder='ViT-S', expr_encoder=None),
    'vision-morph-frozen': _spec('vision-morph-frozen', 'Vision (frozen)', 11, 'uni-modal',
        stage=None, strategy=None, pool=None, learnable_gate=None,
        morph_encoder='ViT-S', expr_encoder=None, freeze_morph=True),
    'expr-token': _spec('expr-token', 'Expression token', 20, 'uni-modal',
        stage=None, strategy=None, pool='token', learnable_gate=None,
        morph_encoder=None, expr_encoder='MLP'),
    'expr-token-expr-frozen': _spec('expr-token-expr-frozen', 'Expression token (frozen)', 21, 'uni-modal',
        stage=None, strategy=None, pool='token', learnable_gate=None,
        morph_encoder=None, expr_encoder='MLP', freeze_expr=True),
    'expr-tile': _spec('expr-tile', 'Expression tile', 30, 'uni-modal',
        stage=None, strategy=None, pool='tile', learnable_gate=None,
        morph_encoder=None, expr_encoder='MLP'),
    'expr-tile-expr-frozen': _spec('expr-tile-expr-frozen', 'Expression tile (frozen)', 31, 'uni-modal',
        stage=None, strategy=None, pool='tile', learnable_gate=None,
        morph_encoder=None, expr_encoder='MLP', freeze_expr=True),
    # ── early fusion ──────────────────────────────────────────────────────────
    'early-fusion': _spec('early-fusion', 'Early fusion', 40, 'multi-modal',
        stage='early', strategy='add', pool='token', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'early-fusion-gate': _spec('early-fusion-gate', 'Early fusion gate', 45, 'multi-modal',
        stage='early', strategy='add', pool='token', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'early-fusion-frozen': _spec('early-fusion-frozen', 'Early fusion (frozen)', 46, 'multi-modal',
        stage='early', strategy='add', pool='token', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
    'early-fusion-gate-frozen': _spec('early-fusion-gate-frozen', 'Early fusion gate (frozen)', 47, 'multi-modal',
        stage='early', strategy='add', pool='token', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
    # ── late fusion token ─────────────────────────────────────────────────────
    'late-fusion-token': _spec('late-fusion-token', 'Late fusion token', 50, 'multi-modal',
        stage='late', strategy='add', pool='token', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'late-fusion-token-gate': _spec('late-fusion-token-gate', 'Late fusion token gate', 55, 'multi-modal',
        stage='late', strategy='add', pool='token', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'late-fusion-token-frozen': _spec('late-fusion-token-frozen', 'Late fusion token (frozen)', 56, 'multi-modal',
        stage='late', strategy='add', pool='token', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
    'late-fusion-token-gate-frozen': _spec('late-fusion-token-gate-frozen', 'Late fusion token gate (frozen)', 57, 'multi-modal',
        stage='late', strategy='add', pool='token', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
    # ── late fusion tile ──────────────────────────────────────────────────────
    'late-fusion-tile': _spec('late-fusion-tile', 'Late fusion tile', 60, 'multi-modal',
        stage='late', strategy='add', pool='tile', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'late-fusion-tile-gate': _spec('late-fusion-tile-gate', 'Late fusion tile gate', 65, 'multi-modal',
        stage='late', strategy='add', pool='tile', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP'),
    'late-fusion-tile-frozen': _spec('late-fusion-tile-frozen', 'Late fusion tile (frozen)', 66, 'multi-modal',
        stage='late', strategy='add', pool='tile', learnable_gate=False,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
    'late-fusion-tile-gate-frozen': _spec('late-fusion-tile-gate-frozen', 'Late fusion tile gate (frozen)', 67, 'multi-modal',
        stage='late', strategy='add', pool='tile', learnable_gate=True,
        morph_encoder='ViT-S', expr_encoder='MLP', freeze_morph=True, freeze_expr=True),
}
validate_slug_specs(SLUG_SPECS)


def load_eval_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text())
    data = raw.pop('data')
    return EvalConfig(data=EvalDataConfig(**data), **raw)


def main(
    eval_cfg: EvalConfig,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    entity: str = DEFAULT_ENTITY,
    wandb_filters: dict | None = None,
) -> None:
    metrics = metrics or list(METRIC_LABELS)
    cache_dir = cache_dir or default_cache_dir()
    output_dir = output_dir or get_managed_paths(eval_cfg.data.name).output_dir / 'figures' / 'eval'

    table = load_project_runs(eval_cfg.project, entity=entity, cache_dir=cache_dir, refresh=eval_cfg.refresh)
    if wandb_filters:
        table = restrict_to_wandb_filter(table, eval_cfg.project, entity=entity, filters=wandb_filters)
    runs, title, name = select_runs(table, eval_cfg=eval_cfg)
    plot_metrics(
        runs,
        specs=SLUG_SPECS,
        metrics=metrics,
        title=title,
        output_prefix=output_dir / name,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--refresh', type=bool, default=None)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--metrics', type=list[str], default=None)
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY)
    parser.add_argument('--wandb-filters', type=dict, default=None)
    return parser


def cli(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    args = namespace.as_dict()
    eval_cfg = load_eval_config(args['config'])
    if args['refresh'] is not None:
        eval_cfg.refresh = args['refresh']
    main(
        eval_cfg=eval_cfg,
        cache_dir=args['cache_dir'],
        output_dir=args['output_dir'],
        metrics=args['metrics'],
        entity=args['entity'],
        wandb_filters=args['wandb_filters'],
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
