from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from xenium_hne_fusion.train.mil import resolve_pretrained_run
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.train.utils import resolve_training_paths
from xenium_hne_fusion.utils.getters import get_managed_paths

load_dotenv(override=True)

CONFIG_PATH = Path("configs/train/beat/mil/expression.yaml")


cfg = MILConfig.from_yaml(CONFIG_PATH)

resolved_run = resolve_pretrained_run(cfg.pretrained)
source_cfg, _ = resolve_training_paths(resolved_run.source_config)

items_path = source_cfg.data.items_path
assert items_path is not None and items_path.exists(), f"items_path not found: {items_path}"

items = pd.read_json(items_path)
assert "sample_id" in items.columns, "sample_id missing from items"

structured_dir = get_managed_paths(source_cfg.data.name).structured_dir
metadata_path = structured_dir / "metadata.parquet"
assert metadata_path.exists(), f"metadata not found: {metadata_path}"

metadata = pd.read_parquet(metadata_path)
metadata.index.name = "sample_id"
metadata = metadata.reset_index()

merged = items.merge(metadata, on="sample_id", how="left")
assert len(merged) == len(items), "merge changed row count"

select = ~merged.isna().any()
merged = merged.loc[:, select]

select = [*items.columns,  '7', '12', '13', '52', '53']
merged = merged[select]

merged['7'].value_counts()

merged['52'].value_counts()
merged['53'].value_counts()

merged['12'].value_counts()
merged['13'].value_counts()

output_path = get_managed_paths(source_cfg.data.name).output_dir / "artifacts/mil" / items_path.stem / "metadata.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(output_path, index=False)
print(f"Saved -> {output_path}")

print(merged.shape)
print(merged.dtypes)
print(merged.head(3))

