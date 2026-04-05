from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from ai4bmr_learn.callbacks.cache import TestCache
from ai4bmr_learn.callbacks.log_model_checkpoint_paths import LogCheckpointPathsCallback
from ai4bmr_learn.callbacks.log_model_stats import LogModelStats
from ai4bmr_learn.callbacks.log_wandb_run_metadata import LogWandbRunMetadataCallback
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import DataLoader

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.models.fusion import FusionModel
from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.models.utils import get_expr_encoder_and_transform, get_morph_encoder_and_transform
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.lit import RegressionLit
from xenium_hne_fusion.train.utils import resolve_training_paths, set_fast_dev_run_settings

TaskName = Literal["gene_prediction", "cell_type_prediction"]

mp.set_sharing_strategy("file_system")
L.seed_everything(0)
torch.set_float32_matmul_precision("high")


def load_panel_config(cfg: Config) -> Config:
    if cfg.data.panel_path is None:
        return cfg

    panel_path = cfg.data.panel_path
    assert panel_path.exists(), f"Panel file not found: {panel_path}"
    panel = yaml.safe_load(panel_path.read_text()) or {}
    if cfg.data.source_panel is None:
        cfg.data.source_panel = panel.get("source_panel")
    if cfg.data.target_panel is None:
        cfg.data.target_panel = panel.get("target_panel")
    return cfg


def resolve_num_source_genes(cfg: Config) -> int | None:
    if cfg.backbone.expr_encoder_name is None:
        return None
    assert cfg.data.source_panel is not None, "cfg.data.source_panel must be set when using an expression encoder"
    return len(cfg.data.source_panel)


def validate_task_config(cfg: Config) -> None:
    assert cfg.task is not None, "cfg.task must be set"
    assert cfg.lit.target_key is not None, "cfg.lit.target_key must be set explicitly"

    if cfg.task == "gene_prediction":
        assert cfg.head.output_dim is None, "cfg.head.output_dim must not be set for task='gene_prediction'"
        assert cfg.lit.target_key == "target", "cfg.lit.target_key must be 'target' for task='gene_prediction'"
        assert cfg.data.target_panel is not None, "cfg.data.target_panel must be set for task='gene_prediction'"
        if cfg.data.source_panel is not None:
            assert set(cfg.data.source_panel).isdisjoint(set(cfg.data.target_panel))
        return

    if cfg.task == "cell_type_prediction":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim must be set for task='cell_type_prediction'"
        assert cfg.lit.target_key != "target", "cfg.lit.target_key must not be 'target' for task='cell_type_prediction'"
        return

    raise ValueError(f"Unknown task: {cfg.task}")


def resolve_num_outputs(cfg: Config) -> int:
    validate_task_config(cfg)
    if cfg.task == "gene_prediction":
        assert cfg.data.target_panel is not None
        return len(cfg.data.target_panel)
    assert cfg.head.output_dim is not None
    return cfg.head.output_dim


def infer_head_input_dim(
    *,
    fusion_stage: str | None,
    fusion_strategy: str | None,
    morph_encoder_dim: int | None,
    expr_encoder_dim: int | None,
) -> int:
    if fusion_stage == "late" and fusion_strategy == "concat":
        assert morph_encoder_dim is not None, "morph_encoder_dim must be set for late concat fusion"
        return morph_encoder_dim * 2
    embed_dim = morph_encoder_dim or expr_encoder_dim
    assert embed_dim is not None, "Could not infer head input dim"
    return embed_dim


def train(cfg: Config, debug: bool | None = None):
    debug = debug if debug is not None else cfg.debug
    if debug or cfg.trainer.fast_dev_run:
        cfg = set_fast_dev_run_settings(cfg)
    cfg, output_dir = resolve_training_paths(cfg)
    cfg = load_panel_config(cfg)
    validate_task_config(cfg)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    num_source_genes = resolve_num_source_genes(cfg)
    num_outputs = resolve_num_outputs(cfg)
    logger.info(f"Training task={cfg.task} with num_outputs={num_outputs} and num_source_genes={num_source_genes}")

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

    lit = RegressionLit(
        backbone=backbone,
        head=head,
        num_outputs=num_outputs,
        batch_key="modalities",
        target_key=cfg.lit.target_key,
        lr_head=cfg.lit.lr_head,
        lr_backbone=cfg.lit.lr_backbone,
        weight_decay=cfg.lit.weight_decay,
        eta=cfg.lit.eta,
        schedule=cfg.lit.schedule,
        num_warmup_epochs=cfg.lit.num_warmup_epochs,
        save_hparams=False,
    )

    dataloader_kws = dict(
        batch_size=cfg.data.batch_size,
        pin_memory=cfg.data.num_workers > 0,
        persistent_workers=cfg.data.num_workers > 0,
        num_workers=cfg.data.num_workers,
    )
    if cfg.data.num_workers > 0 and cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = cfg.data.prefetch_factor

    dataset_kws = dict(
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        panel=cfg.data.source_panel,
        target_panel=cfg.data.target_panel if cfg.task == "gene_prediction" else None,
        include_image=morph_encoder is not None,
        include_expr=expr_encoder is not None,
        image_transform=image_transform,
        expr_transform=expr_transform,
        expr_pool=cfg.data.expr_pool,
        drop_nan_columns=True,
        id_key="id",
    )

    ds_fit = TileDataset(**dataset_kws, split="fit")
    ds_fit.setup()
    ds_val = TileDataset(**dataset_kws, split="val")
    ds_val.setup()
    ds_test = TileDataset(**dataset_kws, split="test")
    ds_test.setup()

    global_batch_size = cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
    dl_fit = DataLoader(ds_fit, shuffle=True, **dataloader_kws)
    dl_val = DataLoader(ds_val, **dataloader_kws)
    dl_test = DataLoader(ds_test, **dataloader_kws)
    log_every_n_steps = min(50, len(dl_fit))

    wb_logger = WandbLogger(
        entity="chuv",
        save_dir=logs_dir,
        **asdict(cfg.wandb),
        config={
            "task": cfg.task,
            "global_batch_size": global_batch_size,
            "num_outputs": num_outputs,
            "num_source_genes": num_source_genes,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            **asdict(cfg),
        },
    )

    monitor = "val/mse_mean"
    mode = "min"
    callbacks = [
        LogModelStats(),
        LogWandbRunMetadataCallback(),
        LogCheckpointPathsCallback(),
        ModelCheckpoint(monitor=monitor, mode=mode, filename="best-{epoch}-{step}", save_last=False),
        ModelCheckpoint(monitor=None, save_last="link"),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor=monitor, mode=mode, patience=10),
        TestCache(exclude_keys=["modalities.image", "loss", "z"]),
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
        log_every_n_steps=log_every_n_steps,
        default_root_dir=output_dir,
        **asdict(cfg.trainer),
    )

    trainer.fit(model=lit, train_dataloaders=dl_fit, val_dataloaders=dl_val)
    if not cfg.trainer.fast_dev_run:
        trainer.test(ckpt_path="best", dataloaders=dl_test)
    wandb.finish()

    return {
        "head": head,
        "backbone": backbone,
        "lit": lit,
        "trainer": trainer,
        "ds_fit": ds_fit,
        "ds_val": ds_val,
    }


def main(cfg: Path, debug: bool = False) -> None:
    train(Config.from_yaml(cfg), debug=debug)
