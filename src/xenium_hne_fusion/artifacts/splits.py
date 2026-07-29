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


def build_single_sample_split(
    items_path: Path,
    sample_id: str,
    fit_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Randomly split one sample's own tiles into fit/val/test (sklearn train_test_split).

    Unlike build_split_metadata_frame (whole samples assigned to one split each), this puts
    tiles from the SAME sample into train, val, and test -- for single-sample experiments.
    """
    from sklearn.model_selection import train_test_split

    items_df = load_items_dataframe(items_path).set_index("id", drop=True)
    sample_items = items_df[items_df["sample_id"] == sample_id]
    assert not sample_items.empty, f"No items for sample_id={sample_id} in {items_path}"
    assert abs(fit_size + val_size + test_size - 1.0) < 1e-9, "fit_size + val_size + test_size must sum to 1"

    # sklearn's internal fancy-indexing chokes on pandas' pyarrow-backed Index (id is `str`
    # dtype, i.e. string[pyarrow]) -- work with a plain numpy array of ids instead.
    ids = sample_items.index.to_numpy()
    fit_ids, rest_ids = train_test_split(ids, train_size=fit_size, random_state=random_state)
    val_ids, test_ids = train_test_split(rest_ids, train_size=val_size / (val_size + test_size), random_state=random_state)

    split_map = {i: "fit" for i in fit_ids} | {i: "val" for i in val_ids} | {i: "test" for i in test_ids}
    sample_items = sample_items.copy()
    sample_items["split"] = sample_items.index.map(split_map)
    return sample_items


def write_single_sample_splits(
    items_path: Path,
    sample_id: str,
    output_dir: Path,
    fit_size: float,
    val_size: float,
    test_size: float,
    n_folds: int,
    random_state: int,
    overwrite: bool = False,
) -> Path:
    """Write n_folds independent random fit/val/test draws for one sample's tiles.

    Each fold is an independent train_test_split call at random_state + i, written to
    outer=<i>.parquet -- same naming convention as write_split_collection's cross-sample folds,
    but here every fold is drawn from the same single sample rather than a different
    sample-to-split assignment.
    """
    split_dir = output_dir / "splits" / sample_id
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_folds):
        out_path = split_dir / f"outer={i}.parquet"
        assert overwrite or not out_path.exists(), f"{out_path} already exists"
        build_single_sample_split(items_path, sample_id, fit_size, val_size, test_size, random_state + i).to_parquet(out_path)
    logger.info(f"Saved {n_folds} single-sample split fold(s) for {sample_id} → {split_dir}")
    return split_dir
