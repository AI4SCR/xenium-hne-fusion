"""Evaluate a trained early-fusion model with expression tokens randomly permuted.

Loads a checkpoint by W&B run_id, enables token permutation on the backbone, and
runs N independent test passes to estimate the permutation baseline (mean ± std).

Usage:
    uv run python scripts/baselines/early-fusion-permuted.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id>
    uv run python scripts/baselines/early-fusion-permuted.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id> --debug true

expr-token-vit: u2l1eisi pbi7jz39 w0y2mwg5 005s9g5a
expr-token: jr7uo9ng yeqi6amk 3pqmw7gh m886lfhu
early-fusion: owzcohia 40utmvmw fif5wuld j5x4suw2
vision: icxsfpqf ua31ay2l cdr1rg1u gnre6bzb

```bash
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G

for RUN_ID in u2l1eisi pbi7jz39 w0y2mwg5 005s9g5a jr7uo9ng yeqi6amk 3pqmw7gh m886lfhu owzcohia 40utmvmw fif5wuld j5x4suw2; do
for RUN_ID in icxsfpqf ua31ay2l cdr1rg1u gnre6bzb; do
sbatch \
    --cpus-per-task=12 \
    --mem=${MEMORY} \
    --gres=gpu:1 \
    --partition=${PARTITION} \
    --time=${TIME} \
    --output=$HOME/logs/%j.out \
    --job-name=cache-${RUN_ID} \
    --wrap="uv run python scribble/cache-results.py --run_id ${RUN_ID}"
done
```

"""
from dataclasses import dataclass
import json
import pandas as pd

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from pathlib import Path

load_dotenv(override=True)

from lightning.pytorch.callbacks import TQDMProgressBar
from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.mil import resolve_pretrained_run
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit, get_target_names
from xenium_hne_fusion.train.utils import prepare_training_config
from ai4bmr_learn.data.splits import Split
from xenium_hne_fusion.utils.getters import get_managed_paths
from ai4bmr_learn.callbacks.cache import PredictionCache


@dataclass
class Config:
    run_id: str
    debug: bool = False
    entity: str = 'chuv'
    project: str = 'xe-hne-fus-cell-v1'


cfg = Config(run_id='w0y2mwg5')
cfg.debug = True


def main(cfg: Config) -> None:
    # cfg = Config(run_id=run_id, debug=debug)

    assert cfg.run_id, "run_id must be set"
    resolved = resolve_pretrained_run(cfg)

    save_dir = get_managed_paths(resolved.source_config.data.name).output_dir / 'results' / cfg.run_id
    logger.info(f'Saving results to {save_dir}')

    resolved_training = prepare_training_config(resolved.source_config)
    logger.info(f"Loaded checkpoint: {resolved.checkpoint_path}")

    target_names = get_target_names(resolved_training)
    logger.info(f"Target names ({len(target_names)}): {target_names[:5]}{'...' if len(target_names) > 5 else ''}")

    dataset_kws = build_supervised_dataset_kws(resolved_training)
    ds_test = TileDataset(**dataset_kws, split=Split.TEST.value)
    ds_test.setup()
    logger.info(f"Test set: {len(ds_test)} tiles")

    # cross-check: config-derived names must match dataset-level panel
    if resolved_training.cfg.task.target == "expression":
        assert ds_test.target_panel is not None
        assert target_names == ds_test.target_panel, (
            f"target_names from config and dataset diverge: "
            f"{set(target_names) ^ set(ds_test.target_panel)}"
        )

    lit = build_supervised_lit(resolved_training, checkpoint_path=resolved.checkpoint_path, target_names=target_names)
    lit.eval()

    dataloader_kws = dict(
        batch_size=resolved_training.cfg.data.batch_size,
        shuffle=False,
        num_workers=min(resolved_training.cfg.data.num_workers, 6),
        pin_memory=resolved_training.cfg.data.num_workers > 0,
        persistent_workers=resolved_training.cfg.data.num_workers > 0,
    )
    if resolved_training.cfg.data.num_workers > 0 and resolved_training.cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = resolved_training.cfg.data.prefetch_factor

    trainer_kws = dict(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
        enable_progress_bar=True,
    )

    if cfg.debug:
        trainer_kws["limit_test_batches"] = 2
        trainer_kws["limit_predict_batches"] = 2
        dataloader_kws["batch_size"] = 2

    dl_test = DataLoader(ds_test, **dataloader_kws)

    cb = [
        TQDMProgressBar(leave=True),
        PredictionCache(save=True, save_dir=save_dir, filestem='predictions', exclude_keys=['modalities'])
    ]
    trainer = L.Trainer(**trainer_kws, callbacks=cb)

    results, = trainer.test(model=lit, dataloaders=dl_test, verbose=True)
    results = pd.DataFrame(results, index=[0])
    results = results.reset_index(drop=True)

    save_path = save_dir / 'results.parquet'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(save_path)

    trainer.predict(model=lit, dataloaders=dl_test, return_predictions=False)
    # torch.save(predictions, save_dir / 'predictions.pt')

    # axes = results.plot.box(column=['test/pearson_mean', 'test/spearman_mean'], by='permute_mode')
    # fig = axes.iloc[0].figure
    # fig.tight_layout()
    # fig.savefig(save_path.with_suffix('.png'))


# %%
if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Config, None)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    d = init.as_dict()
    d.pop("config", None)
    raise SystemExit(main(Config(**d)))
