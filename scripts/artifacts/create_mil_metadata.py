"""Create clinical metadata artifact for MIL training.

Joins cells.json items with structured sample-level metadata, selects the
relevant clinical columns, and writes clinical.parquet to the managed
artifacts directory: DATA_DIR/03_output/<name>/artifacts/mil/<items_stem>/clinical.parquet.

Usage:
    uv run python scripts/artifacts/create_mil_metadata.py --config configs/mil/beat/classification.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from xenium_hne_fusion.train.mil import resolve_pretrained_run
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.train.utils import resolve_training_paths
from xenium_hne_fusion.utils.getters import get_managed_paths


def main(config_path: Path) -> None:
    cfg = MILConfig.from_yaml(config_path)
    resolved_run = resolve_pretrained_run(cfg.pretrained)
    source_cfg, _ = resolve_training_paths(resolved_run.source_config)

    items_path = source_cfg.data.items_path
    assert items_path is not None and items_path.exists(), f"items_path not found: {items_path}"

    items = pd.read_json(items_path)
    assert "sample_id" in items.columns, "sample_id missing from items"

    managed = get_managed_paths(source_cfg.data.name)
    structured_metadata_path = managed.structured_dir / "metadata.parquet"
    assert structured_metadata_path.exists(), f"structured metadata not found: {structured_metadata_path}"

    metadata = pd.read_parquet(structured_metadata_path)
    if "sample_id" not in metadata.columns:
        metadata = metadata.reset_index()
    assert "sample_id" in metadata.columns, "sample_id missing from structured metadata"

    merged = items[["sample_id"]].drop_duplicates().merge(metadata, on="sample_id", how="left")
    assert not merged["sample_id"].duplicated().any(), "duplicate sample_ids after merge"

    # Drop columns with any NaN (samples missing clinical data)
    valid_cols = [col for col in merged.columns if not merged[col].isna().any()]
    merged = merged[valid_cols]

    output_path = managed.output_dir / "artifacts" / "mil" / items_path.stem / "clinical.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(f"Saved clinical artifact -> {output_path} {merged.shape}")
    logger.info(f"Columns: {merged.columns.tolist()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)
