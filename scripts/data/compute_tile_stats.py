"""Compute per-tile statistics from tile-local transcripts.parquet and cells.parquet.

Output:
  DATA_DIR/03_output/<name>/statistics/<items_name>.parquet
  figures/<name>/tile_stats/<items_name>/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger
from tqdm import tqdm

from xenium_hne_fusion.metadata import load_items_dataframe
from xenium_hne_fusion.utils.getters import load_pipeline_config

STAT_COLS = ['num_transcripts', 'num_unique_transcripts', 'num_cells', 'num_unique_cells']


def _compute_item_stats(item: dict, cell_type_col: str) -> dict:
    tile_dir = Path(item['tile_dir'])
    transcripts = pd.read_parquet(tile_dir / 'transcripts.parquet', columns=['feature_name'])
    num_transcripts = len(transcripts)
    num_unique_transcripts = transcripts['feature_name'].nunique()

    num_cells = float('nan')
    num_unique_cells = float('nan')
    cells_path = tile_dir / 'cells.parquet'
    if cells_path.exists():
        cells = pd.read_parquet(cells_path, columns=[cell_type_col])
        num_cells = len(cells)
        num_unique_cells = cells[cell_type_col].nunique()

    return {
        'id': item['id'],
        'num_transcripts': num_transcripts,
        'num_unique_transcripts': num_unique_transcripts,
        'num_cells': num_cells,
        'num_unique_cells': num_unique_cells,
    }


def _format_axis_ticks(ax, axis: str = 'x') -> None:
    """Use 'k' suffix for axes with values >= 1000."""
    the_axis = ax.xaxis if axis == 'x' else ax.yaxis
    lim = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    if max(abs(lim[0]), abs(lim[1])) >= 1000:
        the_axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))


def _resolve_items_path(cfg, items_path: Path | None) -> tuple[Path, str]:
    if items_path is None:
        items_path = cfg.paths.output_dir / 'items' / 'all.json'
    else:
        items_path = Path(items_path)

    assert items_path.suffix == '.json', f'Expected a .json items file, got: {items_path}'
    return items_path, items_path.stem


def _plot_transcript_scatter(stats: pd.DataFrame, output_dir: Path, *, log_axes: bool) -> None:
    scatter = stats[['num_transcripts', 'num_unique_transcripts']].dropna()
    if log_axes:
        scatter = scatter[(scatter['num_transcripts'] > 0) & (scatter['num_unique_transcripts'] > 0)]

    if scatter.empty:
        logger.info(f'Skipping transcript scatter plot with log_axes={log_axes}: no valid rows')
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        scatter['num_transcripts'],
        scatter['num_unique_transcripts'],
        s=8,
        alpha=0.5,
        linewidths=0,
    )
    ax.set_xlabel('num_transcripts')
    ax.set_ylabel('num_unique_transcripts')
    ax.set_title('num_transcripts vs num_unique_transcripts')

    suffix = 'log' if log_axes else 'linear'
    if log_axes:
        ax.set_xscale('log')
        ax.set_yscale('log')

    _format_axis_ticks(ax, 'x')
    _format_axis_ticks(ax, 'y')
    fig.tight_layout()
    output_path = output_dir / f'num_transcripts_vs_num_unique_transcripts_{suffix}.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved diagnostic plot → {output_path}')


def plot_tile_stats(stats: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in STAT_COLS if c in stats.columns and stats[c].notna().any()]

    for col in cols:
        values = stats[col].dropna().values
        n_unique = len(np.unique(values))
        bins = min(50, n_unique)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(col, fontsize=12, fontweight='bold')

        axes[0].hist(values, bins=bins, edgecolor='none')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('count')
        axes[0].set_title('histogram')
        _format_axis_ticks(axes[0], 'x')

        sorted_vals = np.sort(values)
        axes[1].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)))
        for p, alpha in [(0.1, 0.5), (0.25, 0.7)]:
            v = np.quantile(sorted_vals, p)
            axes[1].axvline(v, color='red', linewidth=0.8, alpha=alpha,
                            label=f'p{int(p * 100)}={v:.0f}')
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('cumulative fraction')
        axes[1].set_title('ECDF')
        _format_axis_ticks(axes[1], 'x')

        fig.tight_layout()
        fig.savefig(output_dir / f'{col}.png', dpi=150)
        plt.close(fig)
        logger.info(f'Saved diagnostic plot → {output_dir / col}.png')

    _plot_transcript_scatter(stats, output_dir, log_axes=False)
    _plot_transcript_scatter(stats, output_dir, log_axes=True)


def main(
    dataset: str,
    items_path: Path | None = None,
    cell_type_col: str = 'Level3_grouped',
    config_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    cfg = load_pipeline_config(dataset, config_path)
    items_path, resolved_items_name = _resolve_items_path(cfg, items_path)
    stats_path = cfg.paths.output_dir / 'statistics' / f'{resolved_items_name}.parquet'

    if stats_path.exists() and not overwrite:
        logger.info(f'Statistics already exist: {stats_path}')
        return

    assert items_path.exists(), f'Items not found: {items_path}'
    items_df = load_items_dataframe(items_path)
    items = items_df.to_dict('records')
    logger.info(f'Computing stats for {len(items)} tiles from {items_path}')

    rows = [_compute_item_stats(item, cell_type_col) for item in tqdm(items, desc='Tiles')]
    stats = pd.DataFrame(rows).set_index('id')

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f'Saved statistics → {stats_path}')

    figures_dir = Path('figures') / cfg.processing.name / 'tile_stats' / resolved_items_name
    plot_tile_stats(stats, figures_dir)


if __name__ == '__main__':
    auto_cli(main)
