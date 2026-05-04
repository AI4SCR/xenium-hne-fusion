"""Validate cached MIL prediction bags against the expected tile counts.

For each sample_id, compares the number of items in the source items JSON
against the number of embeddings in the cached .pt file ('z' shape[0]).
Reports missing files and tile-count mismatches.

Usage:
    uv run python scripts/artifacts/validate_prediction_cache.py --config configs/mil/beat/classification.yaml
    uv run python scripts/artifacts/validate_prediction_cache.py --config configs/mil/beat/classification.yaml --debug true
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from xenium_hne_fusion.train.mil import resolve_mil_paths, resolve_pretrained_run
from xenium_hne_fusion.train.mil_config import MILConfig


def main(cfg: MILConfig) -> int:
    resolved_run = resolve_pretrained_run(cfg.pretrained)
    cfg, run_dir = resolve_mil_paths(cfg)

    cache_dir = Path(os.path.expandvars(run_dir / ("debug" if cfg.debug else "predictions")))

    items = json.load(resolved_run.source_config.data.items_path.open())
    expected_counts: dict[str, int] = Counter(i["sample_id"] for i in items)

    missing, mismatched = [], []

    for sample_id, expected in sorted(expected_counts.items()):
        pt_path = cache_dir / f"{sample_id}.pt"

        if not pt_path.exists():
            missing.append(sample_id)
            logger.warning(f"MISSING   {sample_id}  (expected {expected} tiles)")
            continue

        res = torch.load(pt_path, weights_only=False)
        actual = res["z"].shape[0]

        if actual != expected:
            mismatched.append((sample_id, expected, actual))
            logger.warning(f"MISMATCH  {sample_id}  expected={expected}  actual={actual}")
        else:
            logger.info(f"OK        {sample_id}  tiles={actual}")

    logger.info(f"\nSummary: {len(expected_counts)} samples total")
    logger.info(f"  OK:         {len(expected_counts) - len(missing) - len(mismatched)}")
    logger.info(f"  Missing:    {len(missing)}")
    logger.info(f"  Mismatched: {len(mismatched)}")

    return int(bool(missing or mismatched))


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(MILConfig, None)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)

    raise SystemExit(main(MILConfig(**d)))