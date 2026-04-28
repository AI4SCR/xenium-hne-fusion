from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.eval import plotting
from xenium_hne_fusion.eval.experiments import build_plot_output_prefix, resolve_eval_output_dir
from xenium_hne_fusion.eval.plotting import plot_metrics, prepare_plot_table
from xenium_hne_fusion.utils.getters import get_managed_paths


def test_build_plot_output_prefix_uses_stable_dataset_target_name():
    runs = pd.DataFrame([{'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast.yaml'}])
    output_prefix = build_plot_output_prefix(
        runs,
        eval_cfg=EvalConfig(
            project='xe-hne-fus-expr',
            output_dir=Path('/tmp/eval'),
            filters=EvalConfig.Filters(
                target='expression',
                name='hest1k',
                items_path='breast.json',
                metadata_paths=['breast/outer=0-inner=0-seed=0.parquet'],
                panel_paths=['breast-hvg-outer=0-inner=0-seed=0.yaml'],
            ),
        ),
        output_dir=Path('/tmp/eval'),
    )

    assert output_prefix == Path('/tmp/eval/hest1k-expression')


def test_resolve_eval_output_dir_uses_relative_managed_root():
    resolved = resolve_eval_output_dir(
        EvalConfig(
            project='xe-hne-fus-cells',
            output_dir=Path('figures/eval/beat-expression'),
            filters=EvalConfig.Filters(
                target='cell_types',
                name='beat',
                items_path='cells.json',
                metadata_paths=['cells/outer=0-inner=0-seed=0.parquet'],
                panel_paths=['default.yaml'],
            ),
        )
    )

    assert resolved == get_managed_paths('beat').output_dir / 'figures/eval/beat-expression'


def test_resolve_eval_output_dir_keeps_absolute_path():
    resolved = resolve_eval_output_dir(
        EvalConfig(
            project='xe-hne-fus-cells',
            output_dir=Path('/tmp/eval'),
            filters=EvalConfig.Filters(
                target='cell_types',
                name='beat',
                items_path='cells.json',
                metadata_paths=['cells/outer=0-inner=0-seed=0.parquet'],
                panel_paths=['default.yaml'],
            ),
        )
    )

    assert resolved == Path('/tmp/eval')


def test_prepare_plot_table_checks_metrics():
    runs = pd.DataFrame(
        [
            {'run_id': 'r1', 'run_name': 'expr-token', 'config.wandb.name': 'expr-token', 'test/pearson_mean': 0.1},
            {'run_id': 'r2', 'run_name': 'vision', 'config.wandb.name': 'vision', 'test/pearson_mean': 0.2},
        ]
    )

    table = prepare_plot_table(runs, metrics=['test/pearson_mean'])
    assert set(table['model'].tolist()) == {'vision', 'expr-token'}

    with pytest.raises(AssertionError, match='Missing W&B metrics'):
        prepare_plot_table(runs, metrics=['test/spearman_mean'])


def test_prepare_plot_table_warns_and_keeps_latest_duplicate_run(monkeypatch: pytest.MonkeyPatch):
    from xenium_hne_fusion.eval import runs as runs_mod
    warnings = []
    monkeypatch.setattr(runs_mod.logger, 'warning', warnings.append)
    table = pd.DataFrame(
        [
            {
                'run_id': 'old-run',
                'run_name': 'early-fusion',
                'config.wandb.name': 'early-fusion',
                'run_created_at': '2026-01-01T00:00:00Z',
                'config.data.metadata_path': 'default/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': 'expr.yaml',
                'test/pearson_mean': 0.1,
            },
            {
                'run_id': 'new-run',
                'run_name': 'early-fusion',
                'config.wandb.name': 'early-fusion',
                'run_created_at': '2026-01-02T00:00:00Z',
                'config.data.metadata_path': 'default/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': 'expr.yaml',
                'test/pearson_mean': 0.2,
            },
            {
                'run_id': 'other-split',
                'run_name': 'early-fusion',
                'config.wandb.name': 'early-fusion',
                'run_created_at': '2026-01-03T00:00:00Z',
                'config.data.metadata_path': 'default/outer=1-inner=0-seed=0.parquet',
                'config.data.panel_path': 'expr.yaml',
                'test/pearson_mean': 0.3,
            },
        ]
    )

    result = prepare_plot_table(table, metrics=['test/pearson_mean'])

    assert result['run_id'].tolist() == ['new-run', 'other-split']
    assert len(warnings) == 1
    assert 'keeping latest new-run' in warnings[0]
    assert "duplicate run_ids=['old-run', 'new-run']" in warnings[0]


def test_plot_metrics_smoke(tmp_path: Path):
    runs = pd.DataFrame(
        [
            {'run_id': 'r1', 'run_name': 'vision', 'config.wandb.name': 'vision', 'test/pearson_mean': 0.2},
            {'run_id': 'r2', 'run_name': 'vision', 'config.wandb.name': 'vision', 'test/pearson_mean': 0.3},
            {'run_id': 'r3', 'run_name': 'expr-token', 'config.wandb.name': 'expr-token', 'test/pearson_mean': 0.4},
            {'run_id': 'r4', 'run_name': 'expr-token', 'config.wandb.name': 'expr-token', 'test/pearson_mean': 0.5},
        ]
    )

    outputs = plot_metrics(
        runs,
        metrics=['test/pearson_mean'],
        title='Tiny smoke',
        output_prefix=tmp_path / 'tiny',
    )

    assert sorted(path.suffix for path in outputs) == ['.pdf', '.png']
    assert all(path.exists() for path in outputs)


def test_save_runs_csv_writes_split_metadata_suffix_and_scores(tmp_path: Path):
    runs = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'run_name': 'vision',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=0-inner=0-seed=0.parquet',
                'test/pearson_mean': 0.2,
                'test/spearman_mean': 0.3,
            },
        ]
    )

    csv_path = plotting._save_runs_csv(
        runs,
        output_prefix=tmp_path / 'scores',
    )

    table = pd.read_csv(csv_path)
    assert 'model' in table.columns
    assert 'metadata' in table.columns
    assert 'config.data.metadata_path' in table.columns
    assert 'test/pearson_mean' in table.columns
    assert 'test/spearman_mean' in table.columns
    assert table.loc[0, 'metadata'] == 'breast/outer=0-inner=0-seed=0.parquet'
    assert table.loc[0, 'test/pearson_mean'] == pytest.approx(0.2)
    assert table.loc[0, 'test/spearman_mean'] == pytest.approx(0.3)


def test_build_parameter_table_slugs_morph_encoder_names():
    data = pd.DataFrame(
        [
            {'config_id': 'small', 'config.backbone.morph_encoder_name': 'vit_small_patch16_224'},
            {'config_id': 'base', 'config.backbone.morph_encoder_name': 'vit_base_patch16_224.augreg_in21k'},
            {'config_id': 'loki', 'config.backbone.morph_encoder_name': 'loki'},
        ]
    )

    table = plotting._build_parameter_table(
        data,
        ['small', 'base', 'loki'],
        parameter_columns=['config.backbone.morph_encoder_name'],
    )

    assert table.loc['morph_encoder'].to_dict() == {'small': 'ViT-S', 'base': 'ViT-B', 'loki': 'loki'}
