import os
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.utils.getters import get_managed_paths

load_dotenv()

ENTITY = "chuv"

FILTERS = {
    "$and": [
        {"config.data.cache_dir": {"$regex": "default"}},
        {"config.data.items_path": {"$regex": "cells\\.json"}},
        {"config.data.metadata_path": {"$regex": "splits/cells/outer"}},
    ]
}


def main(project: str, name: str) -> None:
    api = wandb.Api()
    rows = []
    for run in api.runs(f"{ENTITY}/{project}", filters=FILTERS):
        run.load(force=True)
        data = run.config.get("data", {})
        cache_dir = data.get("cache_dir") or ""
        items_path = data.get("items_path") or ""
        metadata_path = data.get("metadata_path") or ""
        assert "default" in cache_dir, f"{run.id}: cache_dir={cache_dir!r}"
        assert "cells.json" in items_path, f"{run.id}: items_path={items_path!r}"
        assert "splits/cells" in metadata_path, f"{run.id}: metadata_path={metadata_path!r}"
        rows.append({
            "project_name": project,
            "run_id": run.id,
            "run_name": run.name,
            "items_path": items_path,
            "metadata_path": metadata_path,
            "cache_dir": cache_dir,
        })

    df = pd.DataFrame(rows, columns=["project_name", "run_id", "run_name", "items_path", "metadata_path", "cache_dir"])

    output_dir = get_managed_paths(name).output_dir
    out_path = output_dir / "runs" / f"{project}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.project, args.name))