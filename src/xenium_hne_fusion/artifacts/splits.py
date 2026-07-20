"""Build train/val/test split parquets from a hand-authored sample-assignment YAML."""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from xenium_hne_fusion.artifacts.items import load_items_dataframe

SPLIT_LABEL_MAP = {"train": "fit", "val": "val", "test": "test"}


def load_sample_splits(splits_yaml_path: Path, name: str) -> list[dict[str, list[str]]]:
    all_splits = yaml.safe_load(Path(splits_yaml_path).read_text())
    assert name in all_splits, f"No entry for '{name}' in {splits_yaml_path}"
    return all_splits[name]


def build_split_metadata_frame(items_path: Path, fold: dict[str, list[str]]) -> pd.DataFrame:
    items_df = load_items_dataframe(items_path).set_index("id", drop=True)
    sample_to_split = {
        sample_id: SPLIT_LABEL_MAP[key] for key, sample_ids in fold.items() for sample_id in sample_ids
    }
    missing = set(items_df["sample_id"]) - sample_to_split.keys()
    assert not missing, f"Samples missing from split yaml: {sorted(missing)}"
    items_df["split"] = items_df["sample_id"].map(sample_to_split)
    return items_df


def write_split_collection(
    items_path: Path,
    splits_yaml_path: Path,
    name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    split_dir = output_dir / "splits" / name
    folds = load_sample_splits(splits_yaml_path, name)
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        out_path = split_dir / f"outer={i}.parquet"
        assert overwrite or not out_path.exists(), f"{out_path} already exists"
        build_split_metadata_frame(items_path, fold).to_parquet(out_path)
    logger.info(f"Saved {len(folds)} split fold(s) → {split_dir}")
    return split_dir
