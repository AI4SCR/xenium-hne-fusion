from __future__ import annotations

from pathlib import Path

import marsilea as ma
import marsilea.plotter as mp
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from xenium_hne_fusion.eval.runs import keep_latest_per_group


METRIC_LABELS = {
    'test/pearson_mean': 'Pearson Correlation',
    'test/spearman_mean': 'Spearman Correlation',
}

ANNOTATION_PALETTES = {
    'stage': {'early': '#C5E3C9', 'late': '#AACFDB'},
    'strategy': {'add': '#FAE0B3', 'concat': '#F5D2D2'},
    'pool': {'token': '#BDC3A8', 'tile': '#B7A99F'},
    'learnable_gate': {'False': '#F0F0F0', 'True': '#F2B8A0'},
    'morph_encoder': {'ViT-S': '#ADB2D4', 'ViT-B': '#EEF1DA', 'Loki': '#C7D9DD', 'Phikon': '#D5E5D5'},
    'expr_encoder': {'MLP': '#D7BDE2', 'Geneformer': '#F2B8A0'},
    'freeze_morph': {'False': '#F0F0F0', 'True': '#B0C4DE'},
    'freeze_expr': {'False': '#F0F0F0', 'True': '#B0C4DE'},
}
MODALITY_PALETTE = {'uni-modal': '#A8C8E8', 'multi-modal': '#F5C08A'}
NA_COLOR = '#E0E0E0'

_ANNOTATION_SOURCES = [
    ('stage', 'config.backbone.fusion_stage'),
    ('strategy', 'config.backbone.fusion_strategy'),
    ('pool', 'config.data.expr_pool'),
    ('learnable_gate', 'config.backbone.learnable_gate'),
    ('morph_encoder', 'config.backbone.morph_encoder_name'),
    ('expr_encoder', 'config.backbone.expr_encoder_name'),
    ('freeze_morph', 'config.backbone.freeze_morph_encoder'),
    ('freeze_expr', 'config.backbone.freeze_expr_encoder'),
]
DEFAULT_PARAMETER_COLUMNS = [src_col for _, src_col in _ANNOTATION_SOURCES]
_PARAMETER_LABELS = {src_col: label for label, src_col in _ANNOTATION_SOURCES}
_MORPH_ENCODER_SLUGS = {
    'vit_small_patch16_224': 'ViT-S',
    'vit_base_patch16_224': 'ViT-B',
}


def _relative_metadata_path(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    path = str(value).replace('\\', '/')
    marker = 'splits/'
    idx = path.rfind(marker)
    return path[idx + len(marker):] if idx != -1 else path


def prepare_scores_table(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    missing_metrics = sorted(set(metrics) - set(runs.columns))
    assert not missing_metrics, f'Missing W&B metrics: {missing_metrics}'

    assert 'config.wandb.name' in runs.columns, 'Missing config.wandb.name'
    runs = runs.copy()
    runs['model'] = runs['config.wandb.name'].astype(str)
    runs = keep_latest_per_group(runs)
    if 'config.data.metadata_path' in runs.columns:
        runs['metadata'] = runs['config.data.metadata_path'].apply(_relative_metadata_path)
    return runs.dropna(subset=metrics, how='all').copy()


def prepare_plot_table(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    table = prepare_scores_table(runs, metrics=metrics)
    keep_cols = ['run_id', 'run_name', 'model', 'metadata', *DEFAULT_PARAMETER_COLUMNS, *metrics]
    return table[[c for c in keep_cols if c in table.columns]].copy()


def prepare_metric_plot_table(
    scores: pd.DataFrame,
    *,
    metric: str,
    parameter_columns: list[str],
) -> pd.DataFrame:
    assert metric in scores.columns, f'Missing W&B metric: {metric}'
    missing_columns = sorted(set(parameter_columns) - set(scores.columns))
    assert not missing_columns, f'Missing W&B parameter columns: {missing_columns}'
    keep_cols = ['run_id', 'run_name', 'model', 'metadata', metric, *parameter_columns]
    return scores[[c for c in keep_cols if c in scores.columns]].dropna(subset=[metric]).copy()


def plot_metrics(
    runs: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    output_prefix: Path,
    order_by_name: bool = False,
    parameter_columns: list[str] | None = None,
    color_by_split: bool = False,
) -> list[Path]:
    scores = prepare_scores_table(runs, metrics=metrics)
    if parameter_columns is None:
        parameter_columns = [c for c in DEFAULT_PARAMETER_COLUMNS if c in scores.columns]
    _save_runs_csv(scores, output_prefix=output_prefix)
    outputs = []
    for metric in metrics:
        metric_data = prepare_metric_plot_table(scores, metric=metric, parameter_columns=parameter_columns)
        assert not metric_data.empty, f'No W&B values for {metric}'
        outputs.extend(
            _plot_metric(
                metric_data,
                metric=metric,
                parameter_columns=parameter_columns,
                title=title,
                output_prefix=output_prefix,
                order_by_name=order_by_name,
                color_by_split=color_by_split,
            )
        )
    return outputs


def _save_runs_csv(
    runs: pd.DataFrame,
    *,
    output_prefix: Path,
) -> Path:
    table = runs.copy()
    if 'config.wandb.name' in table.columns:
        table['model'] = table['config.wandb.name'].astype(str)
    if 'metadata' not in table.columns and 'config.data.metadata_path' in table.columns:
        table['metadata'] = table['config.data.metadata_path'].apply(_relative_metadata_path)

    csv_path = output_prefix.with_suffix('.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    logger.info(f'Saved runs CSV ({len(table)} rows) -> {csv_path}')
    return csv_path


def _plot_metric(
    data: pd.DataFrame,
    *,
    metric: str,
    parameter_columns: list[str],
    title: str,
    output_prefix: Path,
    order_by_name: bool,
    color_by_split: bool,
) -> list[Path]:
    models = _ordered_models(data, metric=metric, order_by_name=order_by_name)
    annotations = _build_parameter_table(data, models, parameter_columns=parameter_columns)
    modalities = [_get_modality(data, m) for m in models]
    modality_colors = [MODALITY_PALETTE[mod] for mod in modalities]

    pdat = data.assign(rep=lambda df: df.groupby('model').cumcount()).pivot(
        index='rep', columns='model', values=metric
    ).reindex(columns=models)
    split_data = None
    split_palette = None
    if color_by_split:
        assert 'metadata' in data.columns, 'Missing split metadata'
        assert data['metadata'].notna().all(), 'Missing split metadata'
        split_data = data.assign(rep=lambda df: df.groupby('model').cumcount()).pivot(
            index='rep', columns='model', values='metadata'
        ).reindex(columns=models)
        split_palette = _split_palette(data['metadata'])

    board = ma.ClusterBoard(pdat, height=3.0, margin=0.4)
    board.add_layer(
        mp.Box(pdat, hue=modalities, palette=modality_colors, fill=True, showfliers=False, linewidth=0.7),
        name='boxplot',
    )
    if color_by_split:
        board.add_layer(
            mp.Strip(pdat, hue=split_data, palette=split_palette, jitter=0.25, size=4, alpha=0.75),
            name='stripplot',
        )
    else:
        board.add_layer(mp.Strip(pdat, jitter=0.25, color='black', size=4, alpha=0.75), name='stripplot')
    board.group_cols(models, order=models, spacing=0)
    _add_annotation_rows(board, annotations, models)
    board.add_legends()
    board.add_title(top=f'{title}: {METRIC_LABELS.get(metric, metric)}', fontsize=10, pad=0.5)
    board.render()
    _add_guides(board)
    _add_modality_legend(board, modalities)

    metric_slug = metric.replace('/', '-').replace('_mean', '')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_name(f'{output_prefix.name}-{metric_slug}{suffix}')
        for suffix in ['.pdf', '.png']
    ]
    for output in outputs:
        board.figure.savefig(output, bbox_inches='tight', dpi=300)
        logger.info(f'Saved plot -> {output}')
    plt.close(board.figure)
    return outputs


def _split_palette(metadata: pd.Series) -> dict[str, tuple[float, float, float, float]]:
    splits = sorted(metadata.astype(str).unique())
    cmap = plt.get_cmap('tab20', len(splits))
    return {split: cmap(i) for i, split in enumerate(splits)}


def _ordered_models(data: pd.DataFrame, *, metric: str, order_by_name: bool) -> list[str]:
    models = sorted(data['model'].unique())
    if order_by_name:
        return models
    if metric in data.columns:
        mean_scores = data.groupby('model')[metric].mean()
        return mean_scores.sort_values(ascending=False).index.tolist()
    return models


def _build_parameter_table(data: pd.DataFrame, models: list[str], *, parameter_columns: list[str]) -> pd.DataFrame:
    rows = {}
    for src_col in parameter_columns:
        row_values = {}
        for model in models:
            subset = data.loc[data['model'] == model, src_col]
            val = subset.iloc[0] if not subset.empty else None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = None
            elif src_col == 'config.backbone.morph_encoder_name':
                val = _MORPH_ENCODER_SLUGS.get(str(val).split('.', maxsplit=1)[0], val)
            row_values[model] = val
        rows[_PARAMETER_LABELS.get(src_col, src_col)] = row_values
    return pd.DataFrame(rows).T


def _get_modality(data: pd.DataFrame, model: str) -> str:
    col = 'config.backbone.fusion_strategy'
    if col not in data.columns:
        return 'uni-modal'
    vals = data.loc[data['model'] == model, col]
    return 'multi-modal' if vals.notna().any() and (vals != '').any() else 'uni-modal'


def _add_annotation_rows(board: ma.ClusterBoard, annotations: pd.DataFrame, models: list[str]) -> None:
    for i, row in enumerate(annotations.index):
        palette = ANNOTATION_PALETTES.get(row, {})
        values = [annotations.loc[row, m] for m in models]
        fill_colors = []
        display = []
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                fill_colors.append(NA_COLOR)
                display.append('')
                continue
            fill_colors.append(palette.get(str(value), NA_COLOR))
            display.append(str(value))

        board.add_bottom(
            mp.Chunk(
                display,
                fill_colors=fill_colors,
                label=row,
                align='center',
                props={'fontsize': 6, 'va': 'center_baseline'},
                bordercolor='white',
                borderwidth=0.4,
            ),
            size=0.2,
            pad=0.1 if i == 0 else 0,
        )


def _add_guides(board: ma.ClusterBoard) -> None:
    main_axes = board.get_main_ax()
    if not isinstance(main_axes, (list, tuple)):
        main_axes = [main_axes]
    yticks = main_axes[0].get_yticks()
    for ax in main_axes:
        ax.set_axisbelow(True)
        for y in yticks:
            ax.axhline(y, linestyle='--', linewidth=0.5, alpha=0.7, color='0.6', zorder=0)
        for x in ax.get_xticks():
            ax.axvline(x, linestyle='--', linewidth=0.5, alpha=0.7, color='0.6', zorder=0)


def _add_modality_legend(board: ma.ClusterBoard, modalities: list[str]) -> None:
    main_axes = board.get_main_ax()
    if not isinstance(main_axes, (list, tuple)):
        main_axes = [main_axes]
    handles = [
        Patch(facecolor=color, edgecolor='black', linewidth=0.8, label=modality)
        for modality, color in MODALITY_PALETTE.items()
        if modality in modalities
    ]
    main_axes[-1].legend(
        handles=handles[::-1],
        title='Modality',
        alignment='left',
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        bbox_transform=main_axes[-1].transAxes,
        frameon=False,
    )
