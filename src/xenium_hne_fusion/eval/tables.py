from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from xenium_hne_fusion.eval.plotting import (
    DEFAULT_PARAMETER_COLUMNS,
    _build_parameter_table,
    _config_columns,
    _configuration_ids,
    prepare_scores_table,
)


METRIC_NAMES = {
    'test/spearman_mean': 'spearman',
    'test/pearson_mean': 'pearson',
    'test/mse_mean': 'mse',
}


def prepare_score_latex_table(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
    parameter_columns: list[str] | None = None,
) -> pd.DataFrame:
    scores = prepare_scores_table(runs, metrics=metrics)
    assert 'metadata' in scores.columns, 'Missing split metadata'
    assert scores['metadata'].notna().all(), 'Missing split metadata'

    if parameter_columns is None:
        parameter_columns = [c for c in DEFAULT_PARAMETER_COLUMNS if c in scores.columns]

    config_columns = _config_columns(parameter_columns)
    scores['config_id'] = _configuration_ids(scores, config_columns)
    scores['organ'] = scores['metadata'].apply(_organ_from_metadata)

    group_cols = ['organ', 'config_id']
    grouped = scores.groupby(group_cols, dropna=False, sort=True)
    counts = grouped.size()
    assert (counts >= 2).all(), f'Incomplete split groups: {counts[counts < 2].to_dict()}'

    metric_table = _format_metric_table(grouped, metrics=metrics)
    config_table = _format_config_table(scores, parameter_columns=parameter_columns)
    table = config_table.merge(metric_table, on=group_cols, how='inner')
    assert len(table) == len(metric_table), 'Missing config rows'

    table = table.sort_values(['organ', 'config_id'], kind='stable').drop(columns='config_id')
    return table.reset_index(drop=True)


def save_score_latex_table(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
    output_path: Path,
    parameter_columns: list[str] | None = None,
    caption: str = 'Test scores grouped by organ and model configuration.',
) -> Path:
    table = prepare_score_latex_table(runs, metrics=metrics, parameter_columns=parameter_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(index=False, caption=caption, escape=True, na_rep='-')
    output_path.write_text(latex)
    logger.info(f'Saved score table ({len(table)} rows) -> {output_path}')
    return output_path


def _organ_from_metadata(value: str) -> str:
    path = str(value).replace('\\', '/')
    organ = path.split('/', maxsplit=1)[0]
    assert organ, 'Missing organ'
    return organ


def _format_metric_table(grouped, *, metrics: list[str]) -> pd.DataFrame:
    pieces = []
    for metric in metrics:
        stats = grouped[metric].agg(['mean', 'std'])
        assert stats['std'].notna().all(), f'Missing score std for {metric}'
        name = METRIC_NAMES.get(metric, metric.replace('test/', '').replace('_mean', ''))
        pieces.append(stats.apply(lambda row: f'{row["mean"]:.3f}±{row["std"]:.3f}', axis=1).rename(name))
    return pd.concat(pieces, axis=1).reset_index()


def _format_config_table(scores: pd.DataFrame, *, parameter_columns: list[str]) -> pd.DataFrame:
    configs = sorted(scores['config_id'].unique())
    annotations = _build_parameter_table(scores, configs, parameter_columns=parameter_columns).T
    base = scores[['organ', 'config_id', 'model']].drop_duplicates()
    assert not base.duplicated(['organ', 'config_id']).any(), 'Duplicate config rows'
    return base.merge(annotations.reset_index(names='config_id'), on='config_id', how='left')
