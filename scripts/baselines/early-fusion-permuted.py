"""Evaluate a trained early-fusion model with expression tokens randomly permuted.

Loads a checkpoint by W&B run_id, enables token permutation on the backbone, and
runs N independent test passes to estimate the permutation baseline (mean ± std).

Usage:
    uv run python scripts/baselines/early-fusion-permuted.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id>
    uv run python scripts/baselines/early-fusion-permuted.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id> --debug true
40utmvmw
for run_id in 40utmvmw j5x4suw2 owzcohia fif5wuld; do
    uv run python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/scripts/baselines/early-fusion-permuted.py --config configs/baselines/early-fusion-permuted.yaml --run_id $run_id
done


"""
from __future__ import annotations

import os
from dataclasses import dataclass
import json

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

load_dotenv(override=True)

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.mil import resolve_pretrained_run
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit
from ai4bmr_learn.data.splits import Split
from xenium_hne_fusion.utils.getters import get_managed_paths


@dataclass
class Config:
    entity: str = "chuv"
    project: str = "xe-hne-fus-cell-v1"
    run_id: str = "40utmvmw"
    n_permutations: int = 1
    debug: bool = True

cfg = Config()

def main(cfg: Config) -> None:
    assert cfg.run_id, "run_id must be set"

    resolved = resolve_pretrained_run(cfg)
    source_cfg = resolved.source_config
    logger.info(f"Loaded checkpoint: {resolved.checkpoint_path}")

    lit = build_supervised_lit(source_cfg, checkpoint_path=resolved.checkpoint_path)
    lit.eval()

    # Enable permutation on the loaded backbone
    assert hasattr(lit.backbone, "permute_expr_tokens"), "backbone missing permute_expr_tokens"
    lit.backbone.permute_expr_tokens = True

    dataset_kws = build_supervised_dataset_kws(source_cfg)
    ds_test = TileDataset(**dataset_kws, split=Split.TEST.value)
    ds_test.setup()
    logger.info(f"Test set: {len(ds_test)} tiles")

    dataloader_kws = dict(
        batch_size=source_cfg.data.batch_size,
        shuffle=False,
        num_workers=source_cfg.data.num_workers,
        pin_memory=source_cfg.data.num_workers > 0,
        persistent_workers=source_cfg.data.num_workers > 0,
    )
    if source_cfg.data.num_workers > 0 and source_cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = source_cfg.data.prefetch_factor
    dl_test = DataLoader(ds_test, **dataloader_kws)

    trainer_kws = dict(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
        enable_progress_bar=True,
    )
    if cfg.debug:
        trainer_kws["limit_test_batches"] = 2

    all_results: list[dict] = []
    for i in range(cfg.n_permutations):
        trainer = L.Trainer(**trainer_kws)
        results = trainer.test(model=lit, dataloaders=dl_test, verbose=False)
        assert len(results) == 1
        all_results.append(results[0])
        logger.info(f"Permutation {i + 1}/{cfg.n_permutations}: {results[0]}")

    # Aggregate across permutations
    keys = all_results[0].keys()
    means = {k: torch.tensor([r[k] for r in all_results]).mean().item() for k in keys}
    stds = {k: torch.tensor([r[k] for r in all_results]).std().item() for k in keys}

    logger.info("=== Permuted baseline results ===")
    for k in keys:
        logger.info(f"  {k}: {means[k]:.4f} ± {stds[k]:.4f}")

    output_dir = get_managed_paths(resolved.source_config.data.name).output_dir
    save_path = output_dir / 'baselines' / 'permutation' / f'{cfg.run_id}.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, save_path.open('w'))

# %%
if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Config, None)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)
    raise SystemExit(main(Config(**d)))
