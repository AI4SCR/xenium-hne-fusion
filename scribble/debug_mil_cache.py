from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from xenium_hne_fusion.train.mil import resolve_mil_paths, resolve_pretrained_run, run_prediction_cache
from xenium_hne_fusion.train.mil_config import MILConfig

RUN_IDS = ["lrqyfqta", "dvc1qwis", "9ej3jw59", "2d8h8hr7"]

for run_id in RUN_IDS:
    print(f"\n--- {run_id} ---")
    cfg = MILConfig.from_yaml(Path(f"configs/mil/beat/{run_id}.yaml"))
    cfg.debug = True
    cfg, run_root = resolve_mil_paths(cfg)
    resolved_run = resolve_pretrained_run(cfg.pretrained)
    with tempfile.TemporaryDirectory(prefix=f"mil-debug-{run_id}-") as tmp:
        run_prediction_cache(
            resolved_run=resolved_run,
            cfg=cfg,
            run_root=run_root,
            cache_dir=Path(tmp),
        )
    print(f"OK: {run_id}")
