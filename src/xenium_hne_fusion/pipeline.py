"""Shared pipeline stage utilities for dataset construction."""

import importlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from xenium_hne_fusion.metadata import join_items_with_metadata, load_items_dataframe, save_split_metadata
from xenium_hne_fusion.utils.getters import (
    DEFAULT_CELL_TYPE_COL,
    DEFAULT_SOURCE_ITEMS_NAME,
    ItemsFilterConfig,
    STAT_COLS,
    PipelineConfig,
    apply_filter,
    clear_sample_markers,
    compute_item_stats,
    iter_tile_dirs,
    processed_sample_dir,
    tile_item,
)


def load_ray_module():
    return importlib.import_module("ray")


def wait_for_ray_samples(ray, futures: list[tuple[str, object]]) -> None:
    failed_sample_ids = []
    for sample_id, future in futures:
        try:
            ray.get(future)
        except Exception as exc:
            logger.error(f"{sample_id} failed: {exc}")
            failed_sample_ids.append(sample_id)
    if failed_sample_ids:
        raise RuntimeError(f"Failed samples: {failed_sample_ids}")


def maybe_reset_sample(cfg: PipelineConfig, sample_id: str, overwrite: bool) -> None:
    if not overwrite:
        return
    clear_sample_markers(cfg, sample_id)
    processed_dir = processed_sample_dir(cfg, sample_id)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)


def create_all_items(cfg: PipelineConfig, kernel_size: int = 16, overwrite: bool = False) -> Path:
    items_path = cfg.paths.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    if items_path.exists() and not overwrite:
        logger.info(f"Items already exist: {items_path}")
        return items_path

    sample_dirs = sorted(path for path in cfg.paths.processed_dir.iterdir() if path.is_dir())
    logger.info(f"Building items from {len(sample_dirs)} processed samples")

    items = []
    skipped = []
    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        sample_id = sample_dir.name
        for tile_dir in iter_tile_dirs(sample_dir):
            item = tile_item(tile_dir, sample_id, int(tile_dir.name), kernel_size)
            (items if item is not None else skipped).append(item or tile_dir)

    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps(items, indent=2))
    logger.info(f"Saved {len(items)} items -> {items_path}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} incomplete tile dirs")
    return items_path


def _format_axis_ticks(ax, axis: str = "x") -> None:
    the_axis = ax.xaxis if axis == "x" else ax.yaxis
    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    if max(abs(lim[0]), abs(lim[1])) >= 1000:
        the_axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


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

    _format_axis_ticks(ax, "x")
    _format_axis_ticks(ax, "y")
    fig.tight_layout()
    output_path = output_dir / f"num_transcripts_vs_num_unique_transcripts_{suffix}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved diagnostic plot -> {output_path}")


def plot_tile_stats(stats: pd.DataFrame, output_dir: Path) -> None:
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
        _format_axis_ticks(axes[0], "x")

        sorted_vals = np.sort(values)
        axes[1].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)))
        for p, alpha in [(0.1, 0.5), (0.25, 0.7)]:
            v = np.quantile(sorted_vals, p)
            axes[1].axvline(v, color="red", linewidth=0.8, alpha=alpha, label=f"p{int(p * 100)}={v:.0f}")
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("cumulative fraction")
        axes[1].set_title("ECDF")
        _format_axis_ticks(axes[1], "x")

        fig.tight_layout()
        fig.savefig(output_dir / f"{col}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved diagnostic plot -> {output_dir / col}.png")

    _plot_transcript_scatter(stats, output_dir, log_axes=False)
    _plot_transcript_scatter(stats, output_dir, log_axes=True)


def compute_tile_stats_from_items(
    items_path: Path,
    output_dir: Path,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> Path:
    items_path = Path(items_path)
    output_dir = Path(output_dir)
    stats_path = output_dir / "statistics" / f"{items_path.stem}.parquet"

    if stats_path.exists() and not overwrite:
        logger.info(f"Statistics already exist: {stats_path}")
        return stats_path

    items_df = load_items_dataframe(items_path)
    assert not items_df.empty, f"No items found in {items_path}"
    items = items_df.to_dict("records")
    rows = [compute_item_stats(item, cell_type_col) for item in tqdm(items, desc="Tiles")]
    stats = pd.DataFrame(rows).set_index("id")
    assert list(stats.columns) == STAT_COLS, f"Unexpected stats columns: {stats.columns.tolist()}"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f"Saved statistics -> {stats_path}")

    figures_dir = output_dir / "figures" / "tile_stats" / items_path.stem
    plot_tile_stats(stats, figures_dir)
    return stats_path


def filter_items_from_items_path(
    items_path: Path,
    output_path: Path,
    items_filter_cfg: ItemsFilterConfig,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, int]:
    items_path = Path(items_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info(f"Filtered items already exist: {output_path}")
        return output_path, len(load_items_dataframe(output_path))

    output_dir = output_path.parent.parent
    stats_path = output_dir / "statistics" / f"{items_path.stem}.parquet"
    assert items_path.exists(), f"Source items not found: {items_path}"
    assert stats_path.exists(), f"Statistics not found: {stats_path}"

    items_df = load_items_dataframe(items_path)
    if items_filter_cfg.organs is not None:
        assert metadata_path is not None, "metadata_path is required for organ filtering"
        meta = pd.read_parquet(metadata_path)
        allowed_samples = set(meta.loc[meta.organ.isin(items_filter_cfg.organs), "sample_id"])
        items_df = items_df[items_df["sample_id"].isin(allowed_samples)]

    stats = pd.read_parquet(stats_path)
    kept_ids = set(stats.index[apply_filter(stats, items_filter_cfg)])
    filtered = items_df[items_df["id"].isin(kept_ids)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered.to_dict("records"), indent=2))
    logger.info(f"Filter {items_filter_cfg.name}: {len(items_df)} -> {len(filtered)} tiles")
    logger.info(f"Saved filtered items -> {output_path}")
    return output_path, len(filtered)


def compute_all_tile_stats(
    cfg: PipelineConfig,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> Path:
    items_path = cfg.paths.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    return compute_tile_stats_from_items(items_path, cfg.paths.output_dir, cell_type_col=cell_type_col, overwrite=overwrite)


def create_split_collection(
    cfg: PipelineConfig,
    items_path: Path,
    overwrite: bool = False,
) -> Path:
    split_cfg = cfg.processing.split
    split_dir = cfg.paths.output_dir / "splits" / split_cfg.split_name
    if split_dir.exists() and not overwrite:
        logger.info(f"Split metadata already exists: {split_dir}")
        return split_dir

    joined = join_items_with_metadata(items_path, cfg.paths.processed_dir / "metadata.parquet")
    save_split_metadata(joined, split_dir, split_cfg, overwrite=overwrite)
    return split_dir
