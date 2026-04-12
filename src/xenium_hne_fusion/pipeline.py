"""Shared pipeline stage utilities for dataset construction."""

import importlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.hvg import load_transcript_gene_categories
from xenium_hne_fusion.metadata import build_split_metadata_frame, load_items_dataframe, save_split_metadata
from xenium_hne_fusion.config import FilterConfig, ItemsConfig
from xenium_hne_fusion.utils.getters import (
    DEFAULT_CELL_TYPE_COL,
    DEFAULT_SOURCE_ITEMS_NAME,
    STAT_COLS,
    PipelineConfig,
    apply_filter,
    clear_sample_markers,
    iter_tile_dirs,
    processed_sample_dir,
    select_sample_ids,
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


def _write_tile_stats_summary(items_df: pd.DataFrame, stats: pd.DataFrame, output_path: Path) -> None:
    sample_panels = []
    for sample_id, sample_items in items_df.groupby("sample_id", sort=False):
        del sample_id
        transcripts_path = Path(sample_items.iloc[0]["tile_dir"]) / "transcripts.parquet"
        try:
            panel = set(load_transcript_gene_categories(transcripts_path))
        except AssertionError:
            transcripts = pd.read_parquet(transcripts_path, columns=["feature_name"])
            panel = set(transcripts["feature_name"].dropna().astype(str))
        sample_panels.append(panel)

    panel_sizes = [len(panel) for panel in sample_panels]
    panel_intersection = len(set.intersection(*sample_panels))
    panel_union = len(set.union(*sample_panels))

    num_unique_transcripts = stats["num_unique_transcripts"].dropna()
    summary = {
        "num_tiles": len(items_df),
        "num_samples": items_df["sample_id"].nunique(),
        "num_transcripts": int(stats["num_transcripts"].sum()),
        "num_unique_transcripts_min": int(num_unique_transcripts.min()),
        "num_unique_transcripts_median": float(num_unique_transcripts.median()),
        "num_unique_transcripts_max": int(num_unique_transcripts.max()),
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


def _load_items_df(items_path: Path) -> pd.DataFrame:
    items_df = load_items_dataframe(items_path)
    assert not items_df.empty, f"No items found in {items_path}"
    return items_df


def _compute_expression_stats(
    items_path: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    bootstrap_ds = TileDataset(
        target="expression",
        source_panel=None,
        target_panel=[],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        metadata_path=None,
        id_key="id",
    )
    bootstrap_ds.setup()

    sample_to_items: OrderedDict[str, list[dict]] = OrderedDict()
    for item in bootstrap_ds.items:
        sample_to_items.setdefault(item["sample_id"], []).append(item)

    rows = []
    for sample_id, sample_items in sample_to_items.items():
        feature_universe_path = _resolve_feature_universe_path(Path(sample_items[0]["tile_dir"]))
        feature_universe = feature_universe_path.read_text().splitlines()
        assert feature_universe, f"Empty feature universe: {feature_universe_path}"

        ds = TileDataset(
            target="expression",
            source_panel=None,
            target_panel=feature_universe,
            include_image=False,
            include_expr=False,
            items_path=items_path,
            metadata_path=None,
            id_key="id",
        )
        ds.setup()
        ds.items = [item for item in ds.items if item["sample_id"] == sample_id]
        ds.item_ids = [item["id"] for item in ds.items]

        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        ids = []
        num_transcripts = []
        num_unique_transcripts = []
        for batch in tqdm(dl, desc=f"Tiles[{sample_id}]"):
            batch_target = batch["target"]
            ids.append(list(batch["id"]))
            num_transcripts.append(batch_target.sum(dim=1).tolist())
            num_unique_transcripts.append((batch_target > 0).sum(dim=1).tolist())

        ids = [item_id for batch_ids in ids for item_id in batch_ids]
        num_transcripts = [total for batch_totals in num_transcripts for total in batch_totals]
        num_unique_transcripts = [unique for batch_uniques in num_unique_transcripts for unique in batch_uniques]
        assert len(ids) == len(num_transcripts) == len(num_unique_transcripts), "Mismatched batch stats"

        rows.extend(
            {
                "id": item_id,
                "num_transcripts": int(total),
                "num_unique_transcripts": int(unique),
            }
            for item_id, total, unique in zip(ids, num_transcripts, num_unique_transcripts, strict=True)
        )

    stats = pd.DataFrame(rows).set_index("id")
    assert stats.index.is_unique, "Duplicate item ids in expression stats"
    return stats


def _compute_cell_stats(
    items_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    cell_type_col: str,
) -> pd.DataFrame:
    ds = TileDataset(
        target="cell_types",
        include_image=False,
        include_expr=False,
        items_path=items_path,
        metadata_path=None,
        id_key="id",
        cell_type_col=cell_type_col,
    )
    ds.setup()

    ds.items = [item for item in ds.items if (Path(item["tile_dir"]) / "cells.parquet").exists()]
    ds.item_ids = [item["id"] for item in ds.items]
    if not ds.items:
        return pd.DataFrame(columns=["num_cells", "num_unique_cells"], index=pd.Index([], name="id"))

    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    ids = []
    num_cells = []
    num_unique_cells = []
    for batch in tqdm(dl, desc="Tiles[cells]"):
        batch_target = batch["target"]
        ids.append(list(batch["id"]))
        num_cells.append(batch_target.sum(dim=1).tolist())
        num_unique_cells.append((batch_target > 0).sum(dim=1).tolist())

    ids = [item_id for batch_ids in ids for item_id in batch_ids]
    num_cells = [total for batch_totals in num_cells for total in batch_totals]
    num_unique_cells = [unique for batch_uniques in num_unique_cells for unique in batch_uniques]
    assert len(ids) == len(num_cells) == len(num_unique_cells), "Mismatched cell batch stats"

    stats = pd.DataFrame(
        {
            "id": ids,
            "num_cells": [int(total) for total in num_cells],
            "num_unique_cells": [int(unique) for unique in num_unique_cells],
        }
    ).set_index("id")
    assert stats.index.is_unique, "Duplicate item ids in cell stats"
    return stats


def compute_items_stats(
    items_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 10,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
) -> Path:
    items_path = Path(items_path)
    output_dir = Path(output_dir)
    stats_path = output_dir / "statistics" / f"{items_path.stem}.parquet"

    if stats_path.exists() and not overwrite:
        logger.info(f"Statistics already exist: {stats_path}")
        return stats_path

    items_df = _load_items_df(items_path)
    expression_stats = _compute_expression_stats(
        items_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    cell_stats = _compute_cell_stats(
        items_path,
        batch_size=batch_size,
        num_workers=num_workers,
        cell_type_col=cell_type_col,
    )

    stats = expression_stats.join(cell_stats, how="left")
    stats[["num_cells", "num_unique_cells"]] = stats[["num_cells", "num_unique_cells"]].fillna(0).astype(int)
    stats = stats[STAT_COLS]
    assert list(stats.columns) == STAT_COLS, f"Unexpected stats columns: {stats.columns.tolist()}"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(stats_path)
    logger.info(f"Saved statistics -> {stats_path}")
    _write_tile_stats_summary(items_df, stats, stats_path.with_suffix(".md"))

    figures_dir = output_dir / "figures" / "items" / "stats" / items_path.stem
    plot_items_stats(stats, figures_dir)
    return stats_path


def filter_items(
    items_path: Path,
    output_path: Path,
    items_cfg: ItemsConfig,
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
    loaded_count = len(items_df)
    logger.info(f"Loaded {loaded_count} items from {items_path}")

    if items_cfg.filter.organs is not None:
        assert metadata_path is not None, "metadata_path is required for organ filtering"
        meta = pd.read_parquet(metadata_path)
        before = len(items_df)
        allowed_samples = set(meta.loc[meta.organ.isin(items_cfg.filter.organs), "sample_id"])
        items_df = items_df[items_df["sample_id"].isin(allowed_samples)]
        logger.info(
            f"Organ filter {items_cfg.filter.organs}: {before} -> {len(items_df)} tiles "
            f"({before - len(items_df)} removed)"
        )
    if items_cfg.filter.include_ids is not None or items_cfg.filter.exclude_ids is not None:
        before = len(items_df)
        selected_sample_ids = select_sample_ids(
            sorted(items_df["sample_id"].unique().tolist()),
            FilterConfig(
                include_ids=items_cfg.filter.include_ids,
                exclude_ids=items_cfg.filter.exclude_ids,
            ),
        )
        items_df = items_df[items_df["sample_id"].isin(selected_sample_ids)]
        logger.info(
            f"Sample filter include_ids={items_cfg.filter.include_ids} exclude_ids={items_cfg.filter.exclude_ids}: "
            f"{before} -> {len(items_df)} tiles ({before - len(items_df)} removed)"
        )

    stats = pd.read_parquet(stats_path)
    before = len(items_df)
    kept_ids = set(stats.index[apply_filter(stats, items_cfg)])
    filtered = items_df[items_df["id"].isin(kept_ids)]
    logger.info(
        f"Stats filter {stats_path.name}: {before} -> {len(filtered)} tiles "
        f"({before - len(filtered)} removed)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered.to_dict("records"), indent=2))
    logger.info(f"Filter {items_cfg.name}: {loaded_count} loaded -> {len(filtered)} kept")
    logger.info(f"Saved filtered items -> {output_path}")
    return output_path, len(filtered)


def compute_all_items_stats(
    cfg: PipelineConfig,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> Path:
    items_path = cfg.paths.output_dir / "items" / f"{DEFAULT_SOURCE_ITEMS_NAME}.json"
    return compute_items_stats(
        items_path,
        cfg.paths.output_dir,
        overwrite=overwrite,
        cell_type_col=cell_type_col,
    )


def create_split_collection(
    split_cfg,
    *,
    output_dir: Path,
    processed_dir: Path,
    items_path: Path,
    overwrite: bool = False,
) -> Path:
    split_dir = output_dir / "splits" / split_cfg.name
    if split_dir.exists() and not overwrite:
        logger.info(f"Split metadata already exists: {split_dir}")
        return split_dir

    split_metadata = build_split_metadata_frame(
        items_path,
        split_cfg,
        with_metadata=False,
        sample_metadata_path=processed_dir / "metadata.parquet",
    )
    save_split_metadata(split_metadata, split_dir, split_cfg, overwrite=overwrite)
    return split_dir
