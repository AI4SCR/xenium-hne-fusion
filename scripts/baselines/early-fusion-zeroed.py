"""Evaluate a trained early-fusion model with vision tokens zeroed out.

Loads a checkpoint by W&B run_id, zeros vision tokens on the backbone, and
runs a test pass to measure performance without morphology information.

Usage:
    uv run python scripts/baselines/early-fusion-zeroed.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id>
    uv run python scripts/baselines/early-fusion-zeroed.py \
        --entity chuv --project xe-hne-fus-expr-v0 --run_id <run_id> --debug true

concat: 9cukk99t oub4bwu9 lt9tpohq mx7vx5bi

for run_id in 40utmvmw j5x4suw2 owzcohia fif5wuld; do
    uv run python scripts/baselines/early-fusion-zeroed.py --run_id $run_id
done

for run_id in 9cukk99t oub4bwu9 lt9tpohq mx7vx5bi; do
    uv run python scripts/baselines/early-fusion-zeroed.py --run_id $run_id
done



"""
from dataclasses import dataclass

import pandas as pd

import lightning as L
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from pathlib import Path

load_dotenv(override=True)

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.mil import resolve_pretrained_run
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit
from xenium_hne_fusion.train.utils import prepare_training_config
from ai4bmr_learn.data.splits import Split
from xenium_hne_fusion.utils.getters import get_managed_paths


@dataclass
class Config:
    entity: str = "chuv"
    project: str = "xe-hne-fus-cell-v1"
    run_id: str = "40utmvmw"
    ablation: str = 'zero-out-vision'
    debug: bool = False
    items_path: Path = Path('high_diversity.json')

# cfg = Config()
# cfg.debug = True
# cfg.items_path = Path('high_entropy.json')
# cfg.items_path = Path('high_diversity.json')
# cfg.items_path = Path('low_entropy.json')

def main(cfg: Config) -> None:
    assert cfg.run_id, "run_id must be set"
    resolved = resolve_pretrained_run(cfg)

    save_dir = get_managed_paths(resolved.source_config.data.name).output_dir / 'ablations'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving results to {save_dir}')

    source_cfg = resolved.source_config
    resolved_training = prepare_training_config(source_cfg)
    logger.info(f"Loaded checkpoint: {resolved.checkpoint_path}")

    lit = build_supervised_lit(resolved_training, checkpoint_path=resolved.checkpoint_path)
    lit.eval()

    assert hasattr(lit.backbone, "set_vision_to_zero"), "backbone missing set_vision_to_zero"

    dataset_kws = build_supervised_dataset_kws(resolved_training)
    if cfg.items_path is not None:

        if not cfg.items_path.is_absolute():
            cfg.items_path = get_managed_paths(resolved.source_config.data.name).output_dir / 'items' / cfg.items_path
            assert cfg.items_path.exists()

        dataset_kws['items_path'] = cfg.items_path

    ds_test = TileDataset(**dataset_kws, split=Split.TEST.value)
    ds_test.setup()
    logger.info(f"Test set: {len(ds_test)} tiles")

    dataloader_kws = dict(
        batch_size=source_cfg.data.batch_size,
        shuffle=False,
        num_workers=min(source_cfg.data.num_workers, 6),
        pin_memory=source_cfg.data.num_workers > 0,
        persistent_workers=source_cfg.data.num_workers > 0,
    )
    if source_cfg.data.num_workers > 0 and source_cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = source_cfg.data.prefetch_factor

    trainer_kws = dict(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
        enable_progress_bar=True,
    )

    if cfg.debug:
        trainer_kws["limit_test_batches"] = 2
        dataloader_kws["batch_size"] = 2

    dl_test = DataLoader(ds_test, **dataloader_kws)

    # without ablation
    trainer = L.Trainer(**trainer_kws)
    results, = trainer.test(model=lit, dataloaders=dl_test, verbose=False)
    logger.info(f"Unpermuted: {results}")

    results = pd.DataFrame(results, index=[0])
    results['ablation'] = 'none'

    # with ablation
    lit.backbone.set_vision_to_zero = True
    trainer = L.Trainer(**trainer_kws)
    res, = trainer.test(model=lit, dataloaders=dl_test, verbose=False)
    logger.info(f"Zeroed vision: {res}")

    res = pd.DataFrame(res, index=[0])
    res['ablation'] = cfg.ablation
    results = pd.concat([results, res])

    results = results.reset_index(drop=True)
    results['items'] = str(ds_test.items_path)
    output_dir = get_managed_paths(resolved.source_config.data.name).output_dir
    save_path = output_dir / 'baselines' / 'ablations' / cfg.ablation / cfg.run_id / f'{ds_test.items_path.stem}.parquet'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(save_path)

    ax = results.set_index('ablation')[['test/pearson_mean', 'test/spearman_mean']].plot.bar(rot=0)
    ax.figure.tight_layout()
    ax.figure.savefig(save_path.with_suffix('.png'))

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
