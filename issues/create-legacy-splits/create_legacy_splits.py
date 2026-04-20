"""Create splits matching legacy BEAT split assignments using cells.json items.

Asserts that all sample_ids in cells.json are also present in the legacy
cell_types_filtered.json. Then for each outer fold (outer=X-inner=0-seed=0)
reads the sample_id → split mapping from legacy parquets and applies it to
items in cells.json. Reports per-sample tile count differences vs legacy.
"""

import json
import pandas as pd
from pathlib import Path

LEGACY_DIR = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/xe-hne"
    "/03_output_vprocessed/fusion/splits/beat-v3/512_256_tiles/cell_types_filtered"
)
LEGACY_ITEMS_PATH = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/xe-hne"
    "/03_output_vprocessed/fusion/items/beat-v3/512_256_tiles/cell_types_filtered.json"
)
CELLS_JSON = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0"
    "/03_output/beat/items/cells.json"
)
OUT_DIR = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0"
    "/03_output/beat/splits/cells-legacy"
)
N_OUTER = 4
SPLIT_CATEGORIES = ["fit", "val", "test"]


def load_cells() -> pd.DataFrame:
    with open(CELLS_JSON) as f:
        items = json.load(f)
    df = pd.DataFrame(items).set_index("id")
    assert df.index.is_unique, "cells.json ids are not unique"
    return df


def check_sample_overlap(cells: pd.DataFrame) -> None:
    with open(LEGACY_ITEMS_PATH) as f:
        leg_items = json.load(f)
    leg_sids = {i["sample_id"] for i in leg_items}
    cur_sids = set(cells["sample_id"].unique())
    only_cur = cur_sids - leg_sids
    assert not only_cur, f"cells.json samples not in legacy cell_types_filtered.json: {only_cur}"
    only_leg = leg_sids - cur_sids
    if only_leg:
        print(f"Samples in legacy but not in cells.json ({len(only_leg)}):")
        for s in sorted(only_leg):
            print(f"  {s}")


def legacy_assignment(outer: int) -> dict[str, str]:
    leg = pd.read_parquet(LEGACY_DIR / f"outer={outer}-inner=0-seed=0.parquet")
    per_sample = leg.groupby("sample_id")["split"].nunique()
    assert (per_sample == 1).all(), "Legacy splits are not grouped by sample_id"
    return leg.groupby("sample_id")["split"].first().to_dict()


def legacy_tile_counts(outer: int) -> pd.Series:
    leg = pd.read_parquet(LEGACY_DIR / f"outer={outer}-inner=0-seed=0.parquet")
    return leg.groupby("sample_id").size().rename("legacy")


def create_split(cells: pd.DataFrame, assignment: dict[str, str], outer: int) -> pd.DataFrame:
    cur_sids = set(cells["sample_id"].unique())
    leg_sids = set(assignment.keys())

    unassigned = cur_sids - leg_sids
    if unassigned:
        print(f"  outer={outer}: {len(unassigned)} cells.json samples with no legacy assignment (excluded):")
        for s in sorted(unassigned):
            print(f"    {s}")

    result = cells[cells["sample_id"].isin(leg_sids)].copy()
    result["split"] = result["sample_id"].map(assignment).astype(
        pd.CategoricalDtype(categories=SPLIT_CATEGORIES)
    )
    return result


def report_tile_diff(split_df: pd.DataFrame, outer: int) -> None:
    leg_counts = legacy_tile_counts(outer)
    cur_counts = split_df.groupby("sample_id").size().rename("current")
    merged = pd.concat([cur_counts, leg_counts], axis=1)
    merged["delta"] = merged["current"] - merged["legacy"]
    diff = merged[merged["delta"] != 0].sort_values("delta")
    print(f"  outer={outer}: {len(diff)} samples with different tile counts (vs legacy):")
    if not diff.empty:
        print(diff.to_string())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print(f"Loaded {len(cells)} items from cells.json ({cells['sample_id'].nunique()} samples)")

    check_sample_overlap(cells)

    for outer in range(N_OUTER):
        assignment = legacy_assignment(outer)
        split_df = create_split(cells, assignment, outer)

        counts = split_df["split"].value_counts().to_dict()
        print(f"\nouter={outer}: {counts}")
        report_tile_diff(split_df, outer)

        out_path = OUT_DIR / f"outer={outer}-inner=0-seed=0.parquet"
        split_df.to_parquet(out_path)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
