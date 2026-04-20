from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from scipy.stats import ttest_rel

from xenium_hne_fusion.eval.plotting import prepare_scores_table


DEFAULT_PAIR_KEY = 'config.data.metadata_path'


def paired_t_tests(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
    baseline: str,
    candidates: list[str] | None = None,
    pair_key: str = DEFAULT_PAIR_KEY,
) -> pd.DataFrame:
    scores = prepare_scores_table(runs, metrics=metrics)
    assert pair_key in scores.columns, f'Missing pair key: {pair_key}'
    scores['comparison'] = scores.apply(_comparison_label, axis=1)
    assert baseline in set(scores['comparison']), f'Missing baseline: {baseline}'

    candidates = candidates or sorted(set(scores['comparison']) - {baseline})
    rows = []
    for metric in metrics:
        metric_scores = scores.dropna(subset=[metric])
        _assert_unique_pairs(metric_scores, pair_key=pair_key, metric=metric)
        wide = metric_scores.pivot(index=pair_key, columns='comparison', values=metric)
        assert baseline in wide.columns, f'Missing baseline: {baseline}'

        for candidate in candidates:
            assert candidate in wide.columns, f'Missing candidate: {candidate}'
            paired = wide[[baseline, candidate]].dropna()
            assert len(paired) >= 2, f'Need >=2 paired splits for {candidate} vs {baseline}'
            diff = paired[candidate] - paired[baseline]
            test = ttest_rel(paired[candidate], paired[baseline])
            rows.append(
                {
                    'metric': metric,
                    'baseline': baseline,
                    'candidate': candidate,
                    'n_pairs': len(paired),
                    'mean_baseline': paired[baseline].mean(),
                    'mean_candidate': paired[candidate].mean(),
                    'mean_diff': diff.mean(),
                    't_stat': test.statistic,
                    'p_value': test.pvalue,
                }
            )

    return pd.DataFrame(rows)


def save_paired_t_tests(
    runs: pd.DataFrame,
    *,
    output_path: Path,
    metrics: list[str],
    baseline: str,
    candidates: list[str] | None = None,
    pair_key: str = DEFAULT_PAIR_KEY,
) -> Path:
    table = paired_t_tests(
        runs,
        metrics=metrics,
        baseline=baseline,
        candidates=candidates,
        pair_key=pair_key,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    logger.info(f'Saved paired t-tests ({len(table)} rows) -> {output_path}')
    return output_path


def _comparison_label(row: pd.Series) -> str:
    label = str(row['model'])
    suffixes = []
    if _is_value(row.get('config.backbone.fusion_strategy'), 'concat'):
        suffixes.append('concat')
    if _is_truthy(row.get('config.backbone.learnable_gate')):
        suffixes.append('gate')
    if _is_truthy(row.get('config.backbone.freeze_morph_encoder')):
        suffixes.append('freeze-morph')
    if _is_truthy(row.get('config.backbone.freeze_expr_encoder')):
        suffixes.append('freeze-expr')
    return '-'.join([label, *suffixes])


def _is_value(value, expected: str) -> bool:
    return not _is_missing(value) and str(value) == expected


def _is_truthy(value) -> bool:
    return not _is_missing(value) and str(value).lower() == 'true'


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return bool(pd.isna(value))
    return False


def _assert_unique_pairs(scores: pd.DataFrame, *, pair_key: str, metric: str) -> None:
    duplicates = scores.duplicated([pair_key, 'comparison'], keep=False)
    assert not duplicates.any(), (
        f'Duplicate scores for {metric}; restrict runs or use distinct model names: '
        f'{scores.loc[duplicates, [pair_key, "comparison", "run_id"]].to_dict("records")}'
    )
