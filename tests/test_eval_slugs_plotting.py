from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from scripts.eval.plot_wandb_scores import _build_parser
from xenium_hne_fusion.config import ArtifactsConfig, ItemsConfig, PanelConfig, SplitConfig
from xenium_hne_fusion.eval import plotting
from xenium_hne_fusion.eval.experiments import select_artifact_runs
from xenium_hne_fusion.eval.plotting import plot_metrics, prepare_plot_table


def test_select_artifact_runs_keeps_split_scoped_generated_hvg_panels():
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='breast'),
        split=SplitConfig(name='breast'),
        panel=PanelConfig(
            name='hvg-breast-breast-outer=0-inner=0-seed=0',
            metadata_path=Path('breast/outer=0-inner=0-seed=0.parquet'),
            n_top_genes=16,
            flavor='seurat_v3',
        ),
    )
    runs = pd.DataFrame(
        [
            {
                'run_id': 'outer-0',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast-breast-outer=0-inner=0-seed=0.yaml',
            },
            {
                'run_id': 'outer-2',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast-breast-outer=2-inner=0-seed=0.yaml',
            },
            {
                'run_id': 'wrong-panel-split',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast-breast-outer=0-inner=0-seed=0.yaml',
            },
            {
                'run_id': 'wrong-panel-family',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-default-default-outer=2-inner=0-seed=0.yaml',
            },
            {
                'run_id': 'wrong-split',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/bowel/outer=2-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast-breast-outer=2-inner=0-seed=0.yaml',
            },
            {
                'run_id': 'wrong-items',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/default.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet',
                'config.data.panel_path': '/data/03_output/hest1k/panels/hvg-breast-breast-outer=2-inner=0-seed=0.yaml',
            },
        ]
    )

    selected, title, name = select_artifact_runs(runs, artifacts_cfg=artifacts_cfg, target='expression')

    assert selected['run_id'].tolist() == ['outer-0', 'outer-2']
    assert title == 'HEST1K breast expression'
    assert name == 'hest1k-expression-breast'


def test_select_artifact_runs_handles_static_and_missing_panel_filters():
    static_cfg = ArtifactsConfig(
        name='beat',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default'),
        panel=PanelConfig(name='default'),
    )
    no_panel_cfg = ArtifactsConfig(
        name='beat',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default'),
        panel=None,
    )
    runs = pd.DataFrame(
        [
            {
                'run_id': 'static-panel',
                'config.data.name': 'beat',
                'config.data.items_path': 'items/default.json',
                'config.data.metadata_path': 'splits/default/outer=0-inner=0-seed=0.parquet',
                'config.data.panel_path': 'panels/default.yaml',
            },
            {
                'run_id': 'hvg-panel',
                'config.data.name': 'beat',
                'config.data.items_path': 'items/default.json',
                'config.data.metadata_path': 'splits/default/outer=1-inner=0-seed=0.parquet',
                'config.data.panel_path': 'panels/hvg-default-default-outer=1-inner=0-seed=0.yaml',
            },
        ]
    )

    static_selected, _, _ = select_artifact_runs(runs, artifacts_cfg=static_cfg, target='expression')
    no_panel_selected, _, _ = select_artifact_runs(runs, artifacts_cfg=no_panel_cfg, target='expression')

    assert static_selected['run_id'].tolist() == ['static-panel']
    assert no_panel_selected['run_id'].tolist() == ['static-panel', 'hvg-panel']


def test_select_artifact_runs_keeps_single_split_file_layout():
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='hescape-breast'),
        split=SplitConfig(name='hescape-breast'),
    )
    runs = pd.DataFrame(
        [
            {
                'run_id': 'single-split',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/hescape-breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/hescape-breast.parquet',
                'config.data.panel_path': None,
            },
            {
                'run_id': 'wrong-single-split',
                'config.data.name': 'hest1k',
                'config.data.items_path': '/data/03_output/hest1k/items/hescape-breast.json',
                'config.data.metadata_path': '/data/03_output/hest1k/splits/breast.parquet',
                'config.data.panel_path': None,
            },
        ]
    )

    selected, _, _ = select_artifact_runs(runs, artifacts_cfg=artifacts_cfg, target='expression')

    assert selected['run_id'].tolist() == ['single-split']


def test_eval_plot_cli_uses_artifact_config_scope():
    args = _build_parser().parse_args(
        ['--config', 'configs/artifacts/hest1k/breast.yaml', '--project', 'xe-hne-fus-expr', '--target', 'expression']
    ).as_dict()

    assert args['name'] == 'hest1k'
    assert args['items']['name'] == 'breast'
    assert args['split']['name'] == 'breast'
    assert args['project'] == 'xe-hne-fus-expr'
    assert args['target'] == 'expression'
    assert 'dataset' not in args
    assert 'organ' not in args
    assert 'slugs_path' not in args


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
