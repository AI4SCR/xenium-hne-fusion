"""Compare legacy vs current BEAT splits: sample_ids and per-sample tile counts."""

import pandas as pd
from pathlib import Path

LEGACY_DIR = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/xe-hne"
    "/03_output_vprocessed/fusion/splits/beat-v3/512_256_tiles/cell_types_filtered"
)
CURRENT_DIR = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0"
    "/03_output/beat/splits/cells"
)
N_OUTER = 4


def compare_outer(outer: int) -> None:
    cur = pd.read_parquet(CURRENT_DIR / f"outer={outer}-inner=0-seed=0.parquet")
    leg = pd.read_parquet(LEGACY_DIR / f"outer={outer}-inner=0-seed=0.parquet")

    cur_sids = set(cur["sample_id"].unique())
    leg_sids = set(leg["sample_id"].unique())

    only_leg = sorted(leg_sids - cur_sids)
    only_cur = sorted(cur_sids - leg_sids)

    print(f"\n{'='*60}")
    print(f"outer={outer}")
    print(f"  current:  {len(cur):>7} tiles, {len(cur_sids):>3} samples")
    print(f"  legacy:   {len(leg):>7} tiles, {len(leg_sids):>3} samples")
    print(f"  delta tiles: {len(cur) - len(leg):+d}")

    if only_leg:
        print(f"\n  Samples only in legacy ({len(only_leg)}):")
        for s in only_leg:
            n = leg.loc[leg["sample_id"] == s].shape[0]
            split = leg.loc[leg["sample_id"] == s, "split"].iloc[0]
            print(f"    {s}  tiles={n}  split={split}")

    if only_cur:
        print(f"\n  Samples only in current ({len(only_cur)}):")
        for s in only_cur:
            print(f"    {s}")

    # Per-sample tile count differences for shared samples
    cur_counts = cur.groupby("sample_id").size().rename("current")
    leg_counts = leg.groupby("sample_id").size().rename("legacy")
    merged = pd.concat([cur_counts, leg_counts], axis=1).dropna()
    diff = merged[merged["current"] != merged["legacy"]].copy()
    diff["delta"] = diff["current"] - diff["legacy"]

    print(f"\n  Shared samples with different tile counts: {len(diff)} / {len(merged)}")
    if not diff.empty:
        print(diff.sort_values("delta").to_string())

    # Verify legacy splits are grouped (each sample_id in a single split)
    leg_per_sample = leg.groupby("sample_id")["split"].nunique()
    ungrouped = leg_per_sample[leg_per_sample > 1]
    if ungrouped.empty:
        print("\n  Legacy splits are grouped: each sample_id belongs to exactly one split. ✓")
    else:
        print(f"\n  WARNING: {len(ungrouped)} samples span multiple splits in legacy!")

    # Split assignment comparison for shared samples
    cur_assignment = cur.groupby("sample_id")["split"].first()
    leg_assignment = leg.groupby("sample_id")["split"].first()
    common = cur_assignment.index.intersection(leg_assignment.index)
    mismatch = common[cur_assignment.loc[common] != leg_assignment.loc[common]]
    if mismatch.empty:
        print(f"  Split assignments match for all {len(common)} shared samples. ✓")
    else:
        print(f"\n  Samples with different split assignment ({len(mismatch)}):")
        for s in mismatch:
            print(f"    {s}  current={cur_assignment[s]}  legacy={leg_assignment[s]}")


def main() -> None:
    print("Comparing legacy vs current BEAT splits (outer=X-inner=0-seed=0)\n")
    for outer in range(N_OUTER):
        compare_outer(outer)


if __name__ == "__main__":
    main()
