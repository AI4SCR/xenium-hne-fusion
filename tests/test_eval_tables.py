from pathlib import Path

import pandas as pd
import pytest

from scripts.eval import plot_wandb_score_tables
from xenium_hne_fusion.config import EvalConfig
from xenium_hne_fusion.eval.experiments import select_runs
from xenium_hne_fusion.eval.tables import prepare_score_latex_table


METRICS = ['test/spearman_mean', 'test/pearson_mean', 'test/mse_mean']


def _run(
    *,
    run_id: str,
    organ: str,
    model: str,
    split: int,
    spearman: float,
    pearson: float | None = None,
    mse: float | None = None,
    fusion_stage: str | None = None,
) -> dict:
    return {
        'run_id': run_id,
        'run_name': model,
        'config.wandb.name': model,
        'config.task.target': 'expression',
        'config.data.name': 'hest1k',
        'config.data.items_path': '/data/03_output/hest1k/items/breast.json',
        'config.data.metadata_path': f'/data/03_output/hest1k/splits/{organ}/outer={split}-inner=0-seed=0.parquet',
        'config.data.panel_path': f'/data/03_output/hest1k/panels/hvg-{organ}-outer={split}.yaml',
        'config.backbone.fusion_stage': fusion_stage,
        'test/spearman_mean': spearman,
        'test/pearson_mean': spearman if pearson is None else pearson,
        'test/mse_mean': spearman if mse is None else mse,
    }


def test_prepare_score_latex_table_summarizes_model_scores_by_organ_and_config():
    runs = pd.DataFrame(
        [
            _run(run_id='b0', organ='breast', model='vision', split=0, spearman=0.1, pearson=1.0, mse=2.0),
            _run(run_id='b1', organ='breast', model='vision', split=1, spearman=0.3, pearson=2.0, mse=4.0),
            _run(run_id='b2', organ='breast', model='vision', split=2, spearman=0.5, pearson=3.0, mse=6.0),
            _run(run_id='b3', organ='breast', model='vision', split=3, spearman=0.7, pearson=4.0, mse=8.0),
            _run(run_id='l0', organ='lung', model='vision', split=0, spearman=0.2, pearson=1.0, mse=2.0),
            _run(run_id='l1', organ='lung', model='vision', split=1, spearman=0.4, pearson=2.0, mse=4.0),
        ]
    )

    table = prepare_score_latex_table(runs, metrics=METRICS, parameter_columns=[])

    assert table[['organ', 'model']].to_dict('records') == [
        {'organ': 'breast', 'model': 'vision'},
        {'organ': 'lung', 'model': 'vision'},
    ]
    assert table.loc[0, 'spearman'] == '0.400±0.258'
    assert table.loc[0, 'pearson'] == '2.500±1.291'
    assert table.loc[0, 'mse'] == '5.000±2.582'


def test_prepare_score_latex_table_splits_same_model_by_boxplot_config_id():
    runs = pd.DataFrame(
        [
            _run(run_id='e0', organ='breast', model='fusion', split=0, spearman=0.1, fusion_stage='early'),
            _run(run_id='e1', organ='breast', model='fusion', split=1, spearman=0.3, fusion_stage='early'),
            _run(run_id='l0', organ='breast', model='fusion', split=0, spearman=0.7, fusion_stage='late'),
            _run(run_id='l1', organ='breast', model='fusion', split=1, spearman=0.9, fusion_stage='late'),
        ]
    )

    table = prepare_score_latex_table(
        runs,
        metrics=['test/spearman_mean'],
        parameter_columns=['config.backbone.fusion_stage'],
    )

    assert table[['model', 'stage', 'spearman']].to_dict('records') == [
        {'model': 'fusion', 'stage': 'early', 'spearman': '0.200±0.141'},
        {'model': 'fusion', 'stage': 'late', 'spearman': '0.800±0.141'},
    ]


def test_prepare_score_latex_table_checks_configuration_columns():
    runs = pd.DataFrame(
        [
            _run(run_id='r0', organ='breast', model='vision', split=0, spearman=0.1),
            _run(run_id='r1', organ='breast', model='vision', split=1, spearman=0.2),
        ]
    )

    with pytest.raises(AssertionError, match='Missing configuration columns'):
        prepare_score_latex_table(
            runs,
            metrics=['test/spearman_mean'],
            parameter_columns=['config.backbone.fusion_strategy'],
        )


def test_score_table_cli_writes_latex_from_cached_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runs = pd.DataFrame(
        [
            _run(run_id='r0', organ='breast', model='vision', split=0, spearman=0.1),
            _run(run_id='r1', organ='breast', model='vision', split=1, spearman=0.3),
        ]
    )
    monkeypatch.setattr(plot_wandb_score_tables, 'load_project_runs', lambda *args, **kwargs: runs)

    output_dir = tmp_path / 'tables' / 'eval'
    eval_cfg = EvalConfig(
        project='xe-hne-fus-expr',
        target='expression',
        name='hest1k',
        items_path='breast.json',
        metadata_dir='breast',
        parameter_columns=[],
    )

    output_path = plot_wandb_score_tables.main(eval_cfg=eval_cfg, output_dir=output_dir)

    assert output_path == output_dir / 'hest1k-expression-score-table.tex'
    assert output_path.exists()
    latex = output_path.read_text()
    assert 'organ' in latex
    assert 'model' in latex
    assert 'spearman' in latex
    assert '0.200±0.141' in latex


def test_select_runs_filters_on_logged_task_target():
    runs = pd.DataFrame(
        [
            {
                'run_id': 'expr',
                'config.task.target': 'expression',
                'config.data.name': 'beat',
                'config.data.items_path': '/data/03_output/beat/items/cells.json',
                'config.data.metadata_path': '/data/03_output/beat/splits/cells/outer=0-inner=0-seed=0.parquet',
            },
            {
                'run_id': 'cells',
                'config.task.target': 'cell_types',
                'config.data.name': 'beat',
                'config.data.items_path': '/data/03_output/beat/items/cells.json',
                'config.data.metadata_path': '/data/03_output/beat/splits/cells/outer=0-inner=0-seed=0.parquet',
            },
        ]
    )

    eval_cfg = EvalConfig(
        project='xe-hne-fus-expr-v0',
        target='expression',
        name='beat',
        items_path='cells.json',
        metadata_dir='cells',
    )

    selected, _, _ = select_runs(runs, eval_cfg=eval_cfg)

    assert selected['run_id'].tolist() == ['expr']
