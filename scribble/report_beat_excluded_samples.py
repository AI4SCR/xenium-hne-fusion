"""Report BEAT item and metadata presence for historically excluded sample ids."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli

from xenium_hne_fusion.metadata import load_items_dataframe, normalize_sample_metadata, read_metadata_table


DEFAULT_SAMPLE_IDS = [
    "XE_1JAT.01_HNE_1JAT",
    "XE_1IVR.01_HNE_1IVR_01-1",
    "XE_1FYV.01_HNE_1FYV-01",
    "XE_1GX4.01_HNE_1GX4-01",
]


def _default_data_dir() -> Path:
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "Missing DATA_DIR"
    return Path(data_dir)


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = read_metadata_table(metadata_path)
    if "sample_id" not in metadata.columns:
        assert metadata.index.name == "sample_id", "BEAT metadata must use sample_id index"
        metadata = metadata.reset_index()
    metadata = normalize_sample_metadata(metadata)
    assert metadata["sample_id"].is_unique, "Expected one metadata row per sample_id"
    return metadata


def main(
    items_path: Path | None = None,
    metadata_path: Path | None = None,
    sample_ids: list[str] = DEFAULT_SAMPLE_IDS,
) -> None:
    load_dotenv()

    data_dir = _default_data_dir()
    items_path = Path(items_path) if items_path is not None else data_dir / "03_output" / "beat" / "items" / "all.json"
    metadata_path = (
        Path(metadata_path) if metadata_path is not None else data_dir / "02_processed" / "beat" / "metadata.parquet"
    )
    assert items_path.exists(), f"Missing items JSON: {items_path}"
    assert metadata_path.exists(), f"Missing metadata table: {metadata_path}"

    items_df = load_items_dataframe(items_path)
    metadata_df = _load_metadata(metadata_path)

    observed_item_sample_ids = set(items_df["sample_id"].unique())
    observed_metadata_sample_ids = set(metadata_df["sample_id"].tolist())

    print(f"items_path: {items_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"num_items_total: {len(items_df)}")
    print(f"num_item_samples_total: {items_df['sample_id'].nunique()}")
    print(f"num_metadata_samples_total: {metadata_df['sample_id'].nunique()}")
    print("")
    print("sample_id\titems\tin_items\tin_metadata")

    for sample_id in sample_ids:
        num_items = int((items_df["sample_id"] == sample_id).sum())
        in_items = sample_id in observed_item_sample_ids
        in_metadata = sample_id in observed_metadata_sample_ids
        print(f"{sample_id}\t{num_items}\t{in_items}\t{in_metadata}")

    missing_from_items = sorted(set(sample_ids) - observed_item_sample_ids)
    missing_from_metadata = sorted(set(sample_ids) - observed_metadata_sample_ids)
    items_only_sample_ids = sorted(observed_item_sample_ids - observed_metadata_sample_ids)
    metadata_only_sample_ids = sorted(observed_metadata_sample_ids - observed_item_sample_ids)

    print("")
    print(f"missing_from_items: {missing_from_items}")
    print(f"missing_from_metadata: {missing_from_metadata}")
    print("")
    print(f"num_items_only_sample_ids: {len(items_only_sample_ids)}")
    print(f"num_metadata_only_sample_ids: {len(metadata_only_sample_ids)}")
    print(f"items_only_sample_ids: {items_only_sample_ids}")
    print(f"metadata_only_sample_ids: {metadata_only_sample_ids}")


if __name__ == "__main__":
    auto_cli(main)
