"""Filter the source items list down to items/<name>.json using per-tile statistics."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from xenium_hne_fusion.artifacts.config import ItemsFilterConfig
from xenium_hne_fusion.artifacts.items import apply_filter, load_items_dataframe


def select_sample_ids(
    available_ids: list[str],
    include_ids: list[str] | None,
    exclude_ids: list[str] | None,
) -> list[str]:
    assert include_ids is None or exclude_ids is None, 'include_ids and exclude_ids are mutually exclusive'
    available = sorted(available_ids)
    available_set = set(available)

    if include_ids is not None:
        missing = sorted(set(include_ids) - available_set)
        assert not missing, f'Unknown sample_ids in include_ids: {missing}'
        selected = sorted(include_ids)
    elif exclude_ids is not None:
        exclude_set = set(exclude_ids)
        missing = sorted(exclude_set - available_set)
        assert not missing, f'Unknown sample_ids in exclude_ids: {missing}'
        selected = [sample_id for sample_id in available if sample_id not in exclude_set]
    else:
        selected = available

    assert selected, f'No samples match filter: include_ids={include_ids}, exclude_ids={exclude_ids}'
    return selected


def filter_items(
    items_path: Path,
    output_path: Path,
    stats_path: Path,
    items_cfg: ItemsFilterConfig,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, int]:
    items_path = Path(items_path)
    output_path = Path(output_path)
    stats_path = Path(stats_path)

    if output_path.exists() and not overwrite:
        logger.info(f"Filtered items already exist: {output_path}")
        return output_path, len(load_items_dataframe(output_path))

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
            items_df["sample_id"].unique().tolist(),
            items_cfg.filter.include_ids,
            items_cfg.filter.exclude_ids,
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
