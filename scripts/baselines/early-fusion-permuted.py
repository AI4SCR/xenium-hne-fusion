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
    n_permutations: int = 3
    permute_mode: str = "in-tile"  # 'in-tile' or 'in-batch'
    permute_expr_tokens: bool = False
    permute_vision_tokens: bool = True
    debug: bool = False
    items_path: Path | None = None

# cfg = Config()
# cfg.debug = True
# cfg.items_path = Path('high_entropy.json')
# cfg.items_path = Path('high_diversity.json')
# cfg.items_path = Path('low_entropy.json')

def main(cfg: Config) -> None:
    assert cfg.run_id, "run_id must be set"
    resolved = resolve_pretrained_run(cfg)

    save_dir = get_managed_paths(resolved.source_config.data.name).output_dir / 'baselines' / 'permutation'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving results to {save_dir}')

    source_cfg = resolved.source_config
    logger.info(f"Loaded checkpoint: {resolved.checkpoint_path}")

    lit = build_supervised_lit(source_cfg, checkpoint_path=resolved.checkpoint_path)
    lit.eval()

    # Enable permutation on the loaded backbone
    assert hasattr(lit.backbone, "permute_expr_tokens"), "backbone missing permute_expr_tokens"
    assert hasattr(lit.backbone, "permute_vision_tokens"), "backbone missing permute_vision_tokens"
    assert cfg.permute_mode in ('in-tile', 'in-batch'), f"permute_mode must be 'in-tile' or 'in-batch', got {cfg.permute_mode!r}"
    assert cfg.permute_expr_tokens or cfg.permute_vision_tokens, "at least one of permute_expr_tokens or permute_vision_tokens must be True"

    dataset_kws = build_supervised_dataset_kws(source_cfg)
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

    permute_label = '+'.join(
        [s for s, flag in [('expr', cfg.permute_expr_tokens), ('vision', cfg.permute_vision_tokens)] if flag]
    )

    # without permutation
    lit.backbone.permute_expr_tokens = None
    lit.backbone.permute_vision_tokens = None
    trainer = L.Trainer(**trainer_kws)
    results, = trainer.test(model=lit, dataloaders=dl_test, verbose=False)
    logger.info(f"Unpermuted: {results}")

    results = pd.DataFrame(results, index=[0])
    results['permute_mode'] = 'unpermuted'

    # with permutation
    lit.backbone.permute_expr_tokens = cfg.permute_mode if cfg.permute_expr_tokens else None
    lit.backbone.permute_vision_tokens = cfg.permute_mode if cfg.permute_vision_tokens else None
    for i in range(cfg.n_permutations):
        trainer = L.Trainer(**trainer_kws)
        res, = trainer.test(model=lit, dataloaders=dl_test, verbose=False)
        logger.info(f"[{permute_label}/{cfg.permute_mode}] Permutation {i + 1}/{cfg.n_permutations}: {res}")

        res = pd.DataFrame(res, index=[0])
        res['permute_mode'] = f'{permute_label}/{cfg.permute_mode}'
        results = pd.concat([results, res])

    results = results.reset_index(drop=True)
    results['items'] = str(ds_test.items_path)
    output_dir = get_managed_paths(resolved.source_config.data.name).output_dir
    save_path = output_dir / 'baselines' / 'permutation' / cfg.run_id / f'{permute_label}_{cfg.permute_mode}' / f'{ds_test.items_path.stem}.parquet'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(save_path)

    axes = results.plot.box(column=['test/pearson_mean', 'test/spearman_mean'], by='permute_mode')
    fig = axes.iloc[0].figure
    fig.tight_layout()
    fig.savefig(save_path.with_suffix('.png'))

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
