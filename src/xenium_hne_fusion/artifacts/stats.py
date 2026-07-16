"""Compute and plot per-tile statistics used to filter the source items list."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai4bmr_learn.datasets.items import Items
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenium_hne_fusion.artifacts.items import STAT_COLS, compute_item_stats, load_items_dataframe
from xenium_hne_fusion.processing import load_feature_universe
from xenium_hne_fusion.utils.getters import ManagedPaths


class TileStatisticsDataset(Items):
    """Per-tile transcript/cell counts for artifact statistics.

    Tiles without `transcripts.parquet`/`cells.parquet` count as zero rather than erroring —
    unlike `datasets.tiles.TileDataset`, which assumes a threshold filter has already dropped
    such tiles.
    """

    def __init__(self, *, cell_type_col: str, **kwargs):
        super().__init__(**kwargs)
        self.cell_type_col = cell_type_col

    def __getitem__(self, idx) -> dict:
        return compute_item_stats(self.items[idx], cell_type_col=self.cell_type_col)


@dataclass
class StatsPaths:
    stats: Path
    figures: Path


def default_stats_paths(managed_paths: ManagedPaths, items_path: Path) -> StatsPaths:
    """Default stats/figures paths for `compute_items_stats`, named after `items_path`."""
    return StatsPaths(
        stats=managed_paths.statistics_dir / f"{items_path.stem}.parquet",
        figures=managed_paths.figures_dir / "items" / "stats" / items_path.stem,
    )


def _plot_transcript_scatter(stats: pd.DataFrame, output_dir: Path, *, log_axes: bool) -> None:
    scatter = stats[["num_transcripts", "num_unique_transcripts"]].dropna()
    if log_axes:
        scatter = scatter[(scatter["num_transcripts"] > 0) & (scatter["num_unique_transcripts"] > 0)]

    if scatter.empty:
        logger.info(f"Skipping transcript scatter plot with log_axes={log_axes}: no valid rows")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        scatter["num_transcripts"],
        scatter["num_unique_transcripts"],
        s=8,
        alpha=0.5,
        linewidths=0,
    )
    ax.set_xlabel("num_transcripts")
    ax.set_ylabel("num_unique_transcripts")
    ax.set_title("num_transcripts vs num_unique_transcripts")

    suffix = "log" if log_axes else "linear"
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    output_path = output_dir / f"num_transcripts_vs_num_unique_transcripts_{suffix}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved diagnostic plot -> {output_path}")


def plot_items_stats(stats: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in STAT_COLS if c in stats.columns and stats[c].notna().any()]

    for col in cols:
        values = stats[col].dropna().values
        n_unique = len(np.unique(values))
        bins = min(50, n_unique)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(col, fontsize=12, fontweight="bold")

        axes[0].hist(values, bins=bins, edgecolor="none")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("count")
        axes[0].set_title("histogram")

        sorted_vals = np.sort(values)
        axes[1].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)))
        for p, alpha in [(0.1, 0.5), (0.25, 0.7)]:
            v = np.quantile(sorted_vals, p)
            axes[1].axvline(v, color="red", linewidth=0.8, alpha=alpha, label=f"p{int(p * 100)}={v:.0f}")
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("cumulative fraction")
        axes[1].set_title("ECDF")

        fig.tight_layout()
        fig.savefig(output_dir / f"{col}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved diagnostic plot -> {output_dir / col}.png")

    _plot_transcript_scatter(stats, output_dir, log_axes=False)
    _plot_transcript_scatter(stats, output_dir, log_axes=True)


def _write_tile_stats_summary(items_df: pd.DataFrame, stats: pd.DataFrame, output_path: Path) -> None:
    sample_panels = []
    for sample_id, sample_items in items_df.groupby("sample_id", sort=False):
        del sample_id
        feature_universe_path = _resolve_feature_universe_path(Path(sample_items.iloc[0]["tile_dir"]))
        sample_panels.append(set(load_feature_universe(feature_universe_path)))

    panel_sizes = [len(panel) for panel in sample_panels]
    panel_intersection = len(set.intersection(*sample_panels))
    panel_union = len(set.union(*sample_panels))

    num_transcripts = stats["num_transcripts"].dropna()
    num_unique_transcripts = stats["num_unique_transcripts"].dropna()
    num_unique_cells = stats["num_unique_cells"].dropna()
    summary = {
        "num_tiles": len(items_df),
        "num_samples": items_df["sample_id"].nunique(),
        "num_transcripts": int(stats["num_transcripts"].sum()),
        "num_transcripts_min": int(num_transcripts.min()),
        "num_transcripts_median": float(num_transcripts.median()),
        "num_transcripts_max": int(num_transcripts.max()),
        "num_unique_transcripts_min": int(num_unique_transcripts.min()),
        "num_unique_transcripts_median": float(num_unique_transcripts.median()),
        "num_unique_transcripts_max": int(num_unique_transcripts.max()),
        "num_cells": int(stats["num_cells"].sum()),
        "num_unique_cells_min": int(num_unique_cells.min()),
        "num_unique_cells_median": float(num_unique_cells.median()),
        "num_unique_cells_max": int(num_unique_cells.max()),
        "gene_panel_min": min(panel_sizes),
        "gene_panel_max": max(panel_sizes),
        "gene_panel_intersection": panel_intersection,
        "gene_panel_union": panel_union,
    }

    lines = [f"# {output_path.stem}", ""]
    lines.extend([f"- `{key}`: {value}" for key, value in summary.items()])
    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved statistics summary -> {output_path}")


def _resolve_feature_universe_path(tile_dir: Path) -> Path:
    feature_universe_path = tile_dir.parent.parent / "feature_universe.txt"
    assert feature_universe_path.exists(), f"Missing feature_universe.txt: {feature_universe_path}"
    return feature_universe_path


def compute_items_stats(
    items_path: Path,
    managed_paths: ManagedPaths,
    cell_type_col: str,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 10,
    stats_path: Path | None = None,
    figures_dir: Path | None = None,
) -> Path:
    defaults = default_stats_paths(managed_paths, items_path)
    stats_path = stats_path or defaults.stats
    figures_dir = figures_dir or defaults.figures

    if stats_path.exists() and not overwrite:
        logger.info(f"Statistics already exist: {stats_path}")
        return stats_path

    items_df = load_items_dataframe(items_path)

    ds = TileStatisticsDataset(items_path=items_path, metadata_path=None, id_key="id", cell_type_col=cell_type_col)
    ds.setup()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    stats = pd.concat([pd.DataFrame(batch) for batch in tqdm(dl, desc="Tiles")]).set_index("id")[STAT_COLS]
    assert stats.index.is_unique, "Duplicate item ids in stats"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f"Saved statistics -> {stats_path}")
    _write_tile_stats_summary(items_df, stats, stats_path.with_suffix(".md"))
    plot_items_stats(stats, figures_dir)
    return stats_path
