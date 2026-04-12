import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from scripts.eval.plot_wandb_scores import _build_parser
from xenium_hne_fusion.eval import plotting
from xenium_hne_fusion.eval.experiments import get_eval_task, infer_organ
from xenium_hne_fusion.eval.plotting import plot_metrics, prepare_plot_table
from xenium_hne_fusion.eval.slugs import add_slugs, build_annotation_table, load_slug_specs, ordered_slugs


def test_add_slugs_uses_curated_allowlist_and_rejects_unknown_runs():
    specs = load_slug_specs(Path('configs/eval/slugs.json'))
    runs = pd.DataFrame(
        [
            {'run_name': 'vision', 'config.wandb.name': None},
            {
                'run_name': 'random-wandb-name',
                'config.wandb.name': None,
                'config.backbone.fusion_strategy': 'add',
                'config.backbone.fusion_stage': 'late',
                'config.backbone.morph_encoder_name': 'vit_small_patch16_224',
                'config.backbone.expr_encoder_name': 'mlp',
                'config.backbone.expr_token_pool': None,
                'config.data.expr_pool': 'tile',
            },
        ]
    )

    table = add_slugs(runs, specs)

    assert table['slug'].tolist() == ['vision', 'late-fusion-tile']

    with pytest.raises(AssertionError, match='Runs missing from slugs.json'):
        add_slugs(pd.DataFrame([{'run_name': 'not-allowlisted'}]), specs)


def test_slug_specs_reject_duplicate_orders(tmp_path: Path):
    path = tmp_path / 'slugs.json'
    path.write_text(
        json.dumps(
            {
                'a': {'label': 'A', 'order': 1, 'modality': 'uni-modal', 'stage': None, 'strategy': None, 'pool': None, 'morph_encoder': None, 'expr_encoder': None},
                'b': {'label': 'B', 'order': 1, 'modality': 'uni-modal', 'stage': None, 'strategy': None, 'pool': None, 'morph_encoder': None, 'expr_encoder': None},
            }
        )
    )

    with pytest.raises(AssertionError, match='Duplicate slug orders'):
        load_slug_specs(path)


def test_prepare_plot_table_checks_metrics_and_annotation_order():
    specs = load_slug_specs(Path('configs/eval/slugs.json'))
    runs = pd.DataFrame(
        [
            {'run_id': 'r1', 'run_name': 'expr-token', 'test/pearson_mean': 0.1},
            {'run_id': 'r2', 'run_name': 'vision', 'test/pearson_mean': 0.2},
        ]
    )

    table = prepare_plot_table(runs, specs=specs, metrics=['test/pearson_mean'])
    slugs = ordered_slugs(table, specs)
    annotations = build_annotation_table(specs, slugs)

    assert slugs == ['vision', 'expr-token']
    assert annotations.loc['morph_encoder', 'vision'] == 'ViT-S'
    assert annotations.loc['expr_encoder', 'expr-token'] == 'MLP'

    with pytest.raises(AssertionError, match='Missing W&B metrics'):
        prepare_plot_table(runs, specs=specs, metrics=['test/spearman_mean'])


def test_prepare_plot_table_warns_and_keeps_latest_duplicate_run(monkeypatch: pytest.MonkeyPatch):
    warnings = []
    monkeypatch.setattr(plotting.logger, 'warning', warnings.append)
    specs = load_slug_specs(Path('configs/eval/slugs.json'))
    runs = pd.DataFrame(
        [
            {
                'run_id': 'old-run',
                'run_name': 'early-fusion',
                'run_created_at': '2026-01-01T00:00:00Z',
                'config.data.items_path': 'default.json',
                'config.data.metadata_path': 'default/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': 'default.yaml',
                'test/pearson_mean': 0.1,
            },
            {
                'run_id': 'new-run',
                'run_name': 'early-fusion',
                'run_created_at': '2026-01-02T00:00:00Z',
                'config.data.items_path': 'default.json',
                'config.data.metadata_path': 'default/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': 'default.yaml',
                'test/pearson_mean': 0.2,
            },
            {
                'run_id': 'other-split',
                'run_name': 'early-fusion',
                'run_created_at': '2026-01-03T00:00:00Z',
                'config.data.items_path': 'default.json',
                'config.data.metadata_path': 'default/outer=1-inner=0-seed=0.parquet',
                'config.data.panel_path': 'default.yaml',
                'test/pearson_mean': 0.3,
            },
        ]
    )

    table = prepare_plot_table(runs, specs=specs, metrics=['test/pearson_mean'])

    assert table['run_id'].tolist() == ['new-run', 'other-split']
    assert len(warnings) == 1
    assert 'keeping latest new-run' in warnings[0]
    assert "duplicate run_ids=['old-run', 'new-run']" in warnings[0]


def test_infer_hest1k_organ_from_wandb_tags_or_config():
    assert infer_organ(pd.Series({'tags': '["lung"]'})) == 'lung'
    assert infer_organ(pd.Series({'config.wandb.tags': ['breast']})) == 'breast'
    assert infer_organ(pd.Series({'config.data.metadata_path': 'bowel/outer=0-inner=0-seed=0.parquet'})) == 'bowel'


def test_eval_plot_cli_selects_single_experiment():
    args = _build_parser().parse_args(['--dataset', 'hest1k', '--target', 'expression', '--organ', 'breast']).as_dict()
    task = get_eval_task(dataset=args['dataset'], target=args['target'])

    assert args['dataset'] == 'hest1k'
    assert args['target'] == 'expression'
    assert args['organ'] == 'breast'
    assert task.name == 'hest1k-expression'

    task = get_eval_task(dataset='beat', target='expression')
    assert task.name == 'beat-expression'

    with pytest.raises(AssertionError, match='Unknown eval task'):
        get_eval_task(dataset='unknown', target='expression')


def test_plot_metrics_smoke(tmp_path: Path):
    specs = load_slug_specs(Path('configs/eval/slugs.json'))
    runs = pd.DataFrame(
        [
            {'run_id': 'r1', 'run_name': 'vision', 'test/pearson_mean': 0.2},
            {'run_id': 'r2', 'run_name': 'vision', 'test/pearson_mean': 0.3},
            {'run_id': 'r3', 'run_name': 'expr-token', 'test/pearson_mean': 0.4},
            {'run_id': 'r4', 'run_name': 'expr-token', 'test/pearson_mean': 0.5},
        ]
    )

    outputs = plot_metrics(
        runs,
        specs=specs,
        metrics=['test/pearson_mean'],
        title='Tiny smoke',
        output_prefix=tmp_path / 'tiny',
    )

    assert sorted(path.suffix for path in outputs) == ['.pdf', '.png']
    assert all(path.exists() for path in outputs)
