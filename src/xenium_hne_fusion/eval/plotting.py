from __future__ import annotations

from pathlib import Path

import marsilea as ma
import marsilea.plotter as mp
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from xenium_hne_fusion.eval.slugs import (
    SlugSpec,
    add_slugs,
    build_annotation_table,
    ordered_labels,
    ordered_slugs,
)


METRIC_LABELS = {
    'test/pearson_mean': 'Pearson Correlation',
    'test/spearman_mean': 'Spearman Correlation',
}
ANNOTATION_PALETTES = {
    'stage': {'early': '#C5E3C9', 'late': '#AACFDB'},
    'strategy': {'add': '#FAE0B3', 'concat': '#F5D2D2'},
    'pool': {'token': '#BDC3A8', 'tile': '#B7A99F'},
    'morph_encoder': {'ViT-S': '#ADB2D4', 'ViT-B': '#EEF1DA', 'Loki': '#C7D9DD', 'Phikon': '#D5E5D5'},
    'expr_encoder': {'MLP': '#D7BDE2', 'Geneformer': '#F2B8A0'},
}
MODALITY_PALETTE = {'uni-modal': '#A8C8E8', 'multi-modal': '#F5C08A'}
NA_COLOR = '#E0E0E0'


def prepare_plot_table(
    runs: pd.DataFrame,
    *,
    specs: dict[str, SlugSpec],
    metrics: list[str],
) -> pd.DataFrame:
    missing_metrics = sorted(set(metrics) - set(runs.columns))
    assert not missing_metrics, f'Missing W&B metrics: {missing_metrics}'

    runs = add_slugs(runs, specs)
    keep_cols = ['run_id', 'run_name', 'slug', *metrics]
    data = runs[keep_cols].dropna(subset=metrics, how='all').copy()
    assert not data.empty, 'No runs have the requested W&B metrics'
    return data


def plot_metrics(
    runs: pd.DataFrame,
    *,
    specs: dict[str, SlugSpec],
    metrics: list[str],
    title: str,
    output_prefix: Path,
) -> list[Path]:
    data = prepare_plot_table(runs, specs=specs, metrics=metrics)
    outputs = []
    for metric in metrics:
        metric_data = data.dropna(subset=[metric]).copy()
        assert not metric_data.empty, f'No W&B values for {metric}'
        outputs.extend(
            _plot_metric(metric_data, specs=specs, metric=metric, title=title, output_prefix=output_prefix)
        )
    return outputs


def _plot_metric(
    data: pd.DataFrame,
    *,
    specs: dict[str, SlugSpec],
    metric: str,
    title: str,
    output_prefix: Path,
) -> list[Path]:
    slugs = ordered_slugs(data, specs)
    labels = ordered_labels(slugs, specs)
    pdat = data.assign(rep=lambda df: df.groupby('slug').cumcount()).pivot(
        index='rep',
        columns='slug',
        values=metric,
    )
    pdat = pdat.reindex(columns=slugs)
    pdat.columns = labels

    annotations = build_annotation_table(specs, slugs)
    annotations.columns = labels
    modalities = [specs[slug].modality for slug in slugs]
    modality_colors = [MODALITY_PALETTE[modality] for modality in modalities]

    board = ma.ClusterBoard(pdat, height=3.0, margin=0.4)
    board.add_layer(
        mp.Box(
            pdat,
            hue=modalities,
            palette=modality_colors,
            fill=True,
            showfliers=False,
            linewidth=0.7,
        ),
        name='boxplot',
    )
    board.add_layer(mp.Strip(pdat, jitter=0.25, color='black', size=4, alpha=0.75), name='stripplot')
    board.group_cols(labels, order=labels, spacing=0)
    _add_annotation_rows(board, annotations, labels)
    board.add_legends()
    board.add_title(left=f'{title}: {METRIC_LABELS.get(metric, metric)}', fontsize=10, pad=0.5)
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
    plt.close(board.figure)
    return outputs


def _add_annotation_rows(board: ma.ClusterBoard, annotations: pd.DataFrame, labels: list[str]) -> None:
    for i, row in enumerate(annotations.index):
        palette = ANNOTATION_PALETTES[row]
        values = [annotations.loc[row, label] for label in labels]
        fill_colors = []
        display = []
        for value in values:
            if pd.isna(value):
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
