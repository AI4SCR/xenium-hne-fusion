from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd

import lightning as L
import torch
import torch.multiprocessing as mp
import wandb
from ai4bmr_learn.callbacks.cache import TestCache
from ai4bmr_learn.callbacks.log_model_checkpoint_paths import LogCheckpointPathsCallback
from ai4bmr_learn.callbacks.log_model_stats import LogModelStats
from ai4bmr_learn.callbacks.log_wandb_run_metadata import LogWandbRunMetadataCallback
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import DataLoader

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.models.encoders import log1p_transform
from xenium_hne_fusion.models.fusion import FusionModel
from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.models.utils import get_expr_encoder_and_transform, get_morph_encoder_and_transform
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.lit import ClassificationLit, RegressionLit
from xenium_hne_fusion.train.utils import (
    ResolvedTrainingConfig,
    infer_head_input_dim,
    prepare_training_config,
    resolve_num_outputs,
    resolve_num_source_genes,
    set_fast_dev_run_settings,
    validate_task_config,
)
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL

TaskTarget = Literal["expression", "cell_types"]


def get_target_names(resolved: ResolvedTrainingConfig) -> list[str] | None:
    cfg = resolved.cfg
    if cfg.task.target == "expression":
        return cfg.data.target_panel
    if cfg.task.target == "cell_types":
        items = json.loads(cfg.data.items_path.read_text())
        tile_dir = Path(items[0]["tile_dir"])
        cells = pd.read_parquet(tile_dir / "cells.parquet", columns=[DEFAULT_CELL_TYPE_COL])
        assert cells[DEFAULT_CELL_TYPE_COL].dtype == "category", f"{DEFAULT_CELL_TYPE_COL} must be categorical"
        return sorted(cells[DEFAULT_CELL_TYPE_COL].cat.categories.tolist())
    return None

mp.set_sharing_strategy("file_system")

# Fixes 'cudaErrorInitializationError' by ensuring DataLoader workers start with a clean
# CUDA state. Critical for stability when using multiprocessing on Linux with GPUs.
mp.set_start_method('spawn', force=True)
# Alternatively, clearing cache between train and test can help as well
# torch.cuda.empty_cache()

L.seed_everything(0)
torch.set_float32_matmul_precision("high")


def build_supervised_lit(resolved: ResolvedTrainingConfig, checkpoint_path: str | os.PathLike[str] | None = None, target_names: list[str] | None = None) -> RegressionLit | ClassificationLit:
    cfg = resolved.cfg
    num_source_genes = resolved.num_source_genes
    num_outputs = resolved.num_outputs

    morph_encoder_name = cfg.backbone.morph_encoder_name
    morph_encoder_kws = cfg.backbone.morph_encoder_kws or {}
    expr_encoder_name = cfg.backbone.expr_encoder_name
    expr_encoder_cfg = cfg.backbone.expr_encoder_kws or {}

    assert morph_encoder_name is not None or expr_encoder_name is not None, "At least one encoder must be specified"

    morph_encoder, image_transform, morph_encoder_dim = get_morph_encoder_and_transform(
        morph_encoder_name=morph_encoder_name,
        **morph_encoder_kws,
    )

    expr_encoder = expr_transform = None
    expr_encoder_dim = None
    if expr_encoder_name is not None:
        kws = {**expr_encoder_cfg, "input_dim": num_source_genes}
        expr_encoder, expr_transform, expr_encoder_dim = get_expr_encoder_and_transform(
            expr_encoder_name=expr_encoder_name,
            source_panel=cfg.data.source_panel,
            **kws,
        )

    backbone = FusionModel(
        expr_encoder=expr_encoder,
        morph_encoder=morph_encoder,
        expr_encoder_dim=expr_encoder_dim,
        morph_encoder_dim=morph_encoder_dim,
        fusion_strategy=cfg.backbone.fusion_strategy,
        fusion_stage=cfg.backbone.fusion_stage,
        expr_token_pool=cfg.backbone.expr_token_pool,
        morph_token_pool=cfg.backbone.morph_token_pool,
        global_pool=cfg.backbone.global_pool,
        use_proj=cfg.backbone.use_proj,
        use_modality_embed=cfg.backbone.use_modality_embed,
        learnable_gate=cfg.backbone.learnable_gate,
        morph_key=cfg.backbone.morph_key,
        expr_key=cfg.backbone.expr_key,
        allow_unimodal_routes=cfg.backbone.allow_unimodal_routes,
        pos_embed_layer_name=cfg.backbone.pos_embed_layer_name,
        freeze_morph_encoder=cfg.backbone.freeze_morph_encoder,
        freeze_expr_encoder=cfg.backbone.freeze_expr_encoder,
        set_vision_to_zero=cfg.backbone.set_vision_to_zero,
        normalize_expr_tokens=cfg.backbone.normalize_expr_tokens,
        drop_num_vision_tokens=cfg.backbone.drop_num_vision_tokens,
    )

    embed_dim = infer_head_input_dim(
        fusion_stage=cfg.backbone.fusion_stage,
        fusion_strategy=cfg.backbone.fusion_strategy,
        morph_encoder_dim=morph_encoder_dim,
        expr_encoder_dim=expr_encoder_dim,
    )

    assert cfg.head.num_hidden_layers == 0, "Use only linear probing head for now..."

    head = Head(
        input_dim=embed_dim,
        output_dim=num_outputs,
        hidden_dim=cfg.head.hidden_dim,
        num_hidden_layers=cfg.head.num_hidden_layers,
        dropout=cfg.head.dropout,
    )

    common_kws = dict(
        backbone=backbone,
        head=head,
        batch_key="modalities",
        target_key=cfg.lit.target_key,
        lr_head=cfg.lit.lr_head,
        lr_backbone=cfg.lit.lr_backbone,
        lr_alpha=cfg.lit.lr_alpha,
        weight_decay=cfg.lit.weight_decay,
        eta=cfg.lit.eta,
        schedule=cfg.lit.schedule,
        num_warmup_epochs=cfg.lit.num_warmup_epochs,
        save_hparams=False,
    )

    if cfg.task.type == "classification":
        lit_cls = ClassificationLit
        lit_kws = {**common_kws, "num_classes": num_outputs}
    else:
        lit_cls = RegressionLit
        lit_kws = {**common_kws, "num_outputs": num_outputs, "target_names": target_names}

    if checkpoint_path is not None:
        return lit_cls.load_from_checkpoint(checkpoint_path=checkpoint_path, **lit_kws)
    return lit_cls(**lit_kws)


def build_supervised_dataset_kws(resolved: ResolvedTrainingConfig) -> dict:
    cfg = resolved.cfg

    morph_encoder_name = cfg.backbone.morph_encoder_name
    morph_encoder_kws = cfg.backbone.morph_encoder_kws or {}
    expr_encoder_name = cfg.backbone.expr_encoder_name
    expr_encoder_cfg = cfg.backbone.expr_encoder_kws or {}

    assert morph_encoder_name is not None or expr_encoder_name is not None, "At least one encoder must be specified"

    _, image_transform, _ = get_morph_encoder_and_transform(
        morph_encoder_name=morph_encoder_name,
        **morph_encoder_kws,
    )

    expr_transform = None
    if expr_encoder_name is not None:
        kws = {**expr_encoder_cfg, "input_dim": resolved.num_source_genes}
        _, expr_transform, _ = get_expr_encoder_and_transform(
            expr_encoder_name=expr_encoder_name,
            source_panel=cfg.data.source_panel,
            **kws,
        )

    return dict(
        target=cfg.task.target,
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        source_panel=cfg.data.source_panel,
        target_panel=cfg.data.target_panel if cfg.task.target == "expression" else None,
        include_image=morph_encoder_name is not None,
        include_expr=expr_encoder_name is not None,
        target_transform=log1p_transform,
        image_transform=image_transform,
        expr_transform=expr_transform,
        expr_pool=cfg.data.expr_pool,
        cache_dir=cfg.data.cache_dir,
        drop_nan_columns=True,
        id_key="id",
    )


def train(cfg: Config, debug: bool | None = None, config_path: str | None = None):
    debug = debug if debug is not None else cfg.debug
    if debug or cfg.trainer.fast_dev_run:
        cfg = set_fast_dev_run_settings(cfg)
    resolved = prepare_training_config(cfg)
    cfg = resolved.cfg
    output_dir = resolved.output_dir

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    num_source_genes = resolved.num_source_genes
    num_outputs = resolved.num_outputs
    logger.info(
        f"Training task.target={cfg.task.target} with num_outputs={num_outputs} and num_source_genes={num_source_genes}"
    )

    target_names = get_target_names(resolved)
    lit = build_supervised_lit(resolved, target_names=target_names)

    dataloader_kws = dict(
        batch_size=cfg.data.batch_size,
        pin_memory=cfg.data.num_workers > 0,
        persistent_workers=cfg.data.num_workers > 0,
        num_workers=cfg.data.num_workers,
    )
    if cfg.data.num_workers > 0 and cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = cfg.data.prefetch_factor

    dataset_kws = build_supervised_dataset_kws(resolved)

    if cfg.data.cache_dir is not None:
        # warmup cache: no transforms and no pooling — both are applied post-cache-load per split dataset.
        kws = {**dataset_kws, 'target_transform': None, 'image_transform': None, 'expr_transform': None, 'expr_pool': 'token'}
        ds_all = TileDataset(**kws)
        ds_all.setup()

    ds_fit = TileDataset(**dataset_kws, split="fit")
    ds_fit.setup()
    fit_item = ds_fit[0]
    ds_val = TileDataset(**dataset_kws, split="val")
    ds_val.setup()
    val_item = ds_val[0]
    ds_test = TileDataset(**dataset_kws, split="test")
    ds_test.setup()
    test_item = ds_test[0]

    def print_keys(container: dict):
        for k, v in container.items():
            print(k)
            if isinstance(v, dict):
                print_keys(v)

    print_keys(fit_item)
    print_keys(val_item)
    print_keys(test_item)

    if cfg.wandb.name in ['expr-token', 'expr-tile', 'expr-token-vit']:
        assert 'image' not in fit_item['modalities']
        assert 'image' not in val_item['modalities']
        assert 'image' not in test_item['modalities']
    elif cfg.wandb.name in ['early-fusion-vit', 'early-fusion', 'late-fusion-tile', 'late-fusion-token', 'late-fusion-token-vit']:
        assert 'image' in fit_item['modalities']
        assert 'image' in val_item['modalities']
        assert 'image' in test_item['modalities']
        assert 'expr_tokens' in fit_item['modalities']
        assert 'expr_tokens' in val_item['modalities']
        assert 'expr_tokens' in test_item['modalities']
    elif cfg.wandb.name in ['vision']:
        assert 'expr_tokens' not in fit_item['modalities']
        assert 'expr_tokens' not in val_item['modalities']
        assert 'expr_tokens' not in test_item['modalities']
    else:
        raise ValueError(f"Unknown wandb.name: {cfg.wandb.name}")

    global_batch_size = cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
    dl_fit = DataLoader(ds_fit, shuffle=True, **dataloader_kws)
    dl_val = DataLoader(ds_val, **dataloader_kws)
    dl_test = DataLoader(ds_test, **dataloader_kws)
    cfg.trainer.log_every_n_steps = min(50, len(dl_fit))

    wb_logger = WandbLogger(
        entity="chuv",
        save_dir=logs_dir,
        **asdict(cfg.wandb),
        config={
            "task": cfg.task.target,
            "global_batch_size": global_batch_size,
            "num_outputs": num_outputs,
            "num_source_genes": num_source_genes,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "config_path": config_path,
            **asdict(cfg),
        },
    )

    if cfg.task.type == "classification":
        monitor, mode = "val/accuracy", "max"
    else:
        monitor, mode = "val/mse_mean", "min"
    callbacks = [
        LogModelStats(),
        LogWandbRunMetadataCallback(),
        LogCheckpointPathsCallback(),
        ModelCheckpoint(monitor=monitor, mode=mode, filename="best-{epoch}-{step}", save_last=False),
        ModelCheckpoint(monitor=None, save_last="link"),
        LearningRateMonitor(logging_interval="epoch"),
        # EarlyStopping(monitor=monitor, mode=mode, patience=15),
        TestCache(exclude_keys=["modalities", "loss", "z"]),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        precision="16-mixed",
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=wb_logger,
        callbacks=callbacks,
        val_check_interval=None,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=None,
        default_root_dir=output_dir,
        **asdict(cfg.trainer),
    )

    trainer.fit(model=lit, train_dataloaders=dl_fit, val_dataloaders=dl_val)
    if not cfg.trainer.fast_dev_run:
        torch.cuda.empty_cache()
        trainer.test(ckpt_path="best", dataloaders=dl_test)
    wandb.finish()

    return {
        "lit": lit,
        "trainer": trainer,
        "ds_fit": ds_fit,
        "ds_val": ds_val,
    }


def main(cfg: Config, debug: bool | None = None, config_path: str | None = None) -> None:
    train(cfg, debug=debug, config_path=config_path)
