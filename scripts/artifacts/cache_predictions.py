"""Extract and cache tile embeddings for MIL training.

Runs the pretrained model referenced in the MIL config over all tiles (all splits),
groups embeddings by sample_id, and writes bags.json plus one <sample_id>.pt per
bag to the managed MIL run directory.

Must be run before scripts/train/mil.py.

Usage:
    uv run python scripts/artifacts/cache_predictions.py --config configs/mil/beat/classification.yaml
    uv run python scripts/artifacts/cache_predictions.py --config configs/mil/beat/classification.yaml --debug true
    uv run python scripts/artifacts/cache_predictions.py --config configs/mil/beat/classification.yaml --overwrite true
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from jsonargparse import auto_parser
from tqdm import tqdm
from loguru import logger

load_dotenv(override=True)

from xenium_hne_fusion.train.mil import resolve_mil_paths, resolve_pretrained_run
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit, set_fast_dev_run_settings
from torch.utils.data import DataLoader
from ai4bmr_learn.callbacks.cache import PredictionCache
import lightning as L


def main(cfg: MILConfig, overwrite: bool = False) -> None:
    resolved_run = resolve_pretrained_run(cfg.pretrained)

    cfg, run_dir = resolve_mil_paths(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.debug:
        set_fast_dev_run_settings(cfg)
        set_fast_dev_run_settings(resolved_run.source_config)
        cache_dir = run_dir / 'debug'
    else:
        cache_dir = run_dir / 'predictions'
    cache_dir = Path(os.path.expandvars(cache_dir))

    lit = build_supervised_lit(resolved_run.source_config, checkpoint_path=resolved_run.checkpoint_path)
    lit.eval()

    dataset_kws = build_supervised_dataset_kws(resolved_run.source_config)

    dataloader_kws = dict(
        batch_size=resolved_run.source_config.data.batch_size,
        shuffle=False,
        pin_memory=resolved_run.source_config.data.num_workers > 0,
        persistent_workers=resolved_run.source_config.data.num_workers > 0,
        num_workers=resolved_run.source_config.data.num_workers,
    )
    if resolved_run.source_config.data.num_workers > 0 and resolved_run.source_config.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = resolved_run.source_config.data.prefetch_factor

    trainer_kws = dict(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
    )
    if cfg.debug:
        trainer_kws["limit_predict_batches"] = 2

    items = json.load(resolved_run.source_config.data.items_path.open())
    sample_ids = {i['sample_id'] for i in items}

    ds = TileDataset(**dataset_kws, split=None)
    ds.setup()

    bag_items = []
    for sample_id in tqdm(sample_ids):
        output_path = cache_dir / f'{sample_id}.pt'

        if output_path.exists() and not overwrite:
            logger.info(f'Skipping {sample_id} — cache exists at {output_path}')
            bag_items.append({'sample_id': sample_id, 'path': str(output_path)})
            continue

        logger.info(f'Caching predictions for {sample_id}')
        ds.items = list(filter(lambda x: x['sample_id'] == sample_id, items))
        dl = DataLoader(ds, **dataloader_kws)

        prediction_cache = PredictionCache(
            save=True,
            save_dir=cache_dir,
            filestem=sample_id,
            include_keys=["z", "id", "sample_id"],
        )
        trainer = L.Trainer(**trainer_kws, callbacks=[prediction_cache])
        trainer.predict(model=lit, dataloaders=dl, return_predictions=False)

        assert output_path.exists(), f'{output_path} does not exist'

        res = torch.load(output_path)

        sink = res[0]
        for item in res[1:]:
            for k in item.keys():
                if isinstance(item[k], torch.Tensor):
                    sink[k] = torch.vstack((sink[k], item[k]))
                elif isinstance(item[k], list):
                    sink[k].extend(item[k])
                else:
                    raise ValueError('unsupported dtype')

        torch.save(sink, output_path)
        bag_items.append({'sample_id': sample_id, 'path': str(output_path)})

    bags_path = cache_dir / "bags.json"
    with bags_path.open("w", encoding="utf-8") as f:
        json.dump(bag_items, f, indent=4)

    logger.info(f'Wrote {len(bag_items)} bags to {bags_path}')


if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_class_arguments(MILConfig, None)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)
    overwrite = d.pop("overwrite", False)

    raise SystemExit(main(MILConfig(**d), overwrite=overwrite))