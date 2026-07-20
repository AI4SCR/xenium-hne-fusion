"""One-off: export sample IDs per train/val/test for every outer split (inner=0 only) of each item set."""

import re
from pathlib import Path

import pandas as pd
import yaml

SPLITS_DIR = Path(
    "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v4/03_output/owkin/splits"
)
OUT_PATH = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/splits/owkin.yaml")

SPLIT_KEY_MAP = {"fit": "train", "val": "val", "test": "test"}
OUTER_INNER0_RE = re.compile(r"^outer=(\d+)-inner=0-seed=0\.parquet$")


def main() -> int:
    result = {}
    for item_dir in sorted(p for p in SPLITS_DIR.iterdir() if p.is_dir()):
        outer_splits = []
        for split_path in sorted(
            item_dir.glob("outer=*-inner=0-seed=0.parquet"),
            key=lambda p: int(OUTER_INNER0_RE.match(p.name).group(1)),
        ):
            df = pd.read_parquet(split_path, columns=["sample_id", "split"])
            assert set(df["split"].unique()) <= SPLIT_KEY_MAP.keys(), f"unexpected split values in {split_path}"
            outer_splits.append(
                {
                    SPLIT_KEY_MAP[split]: sorted(group["sample_id"].unique().tolist())
                    for split, group in df.groupby("split", observed=True)
                }
            )
        if outer_splits:
            result[item_dir.name] = outer_splits

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
