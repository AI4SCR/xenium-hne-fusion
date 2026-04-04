from dotenv import load_dotenv

assert load_dotenv(override=True)

import wandb
import os
import lightning as L
import torch
import yaml
from ai4bmr_learn.callbacks.log_model_stats import LogModelStats
from ai4bmr_learn.callbacks.log_wandb_run_metadata import LogWandbRunMetadataCallback
from ai4bmr_learn.callbacks.log_model_checkpoint_paths import LogCheckpointPathsCallback
from ai4bmr_learn.callbacks.cache import TestCache
from xenium_hne_fusion.models.fusion import FusionModel
from xenium_hne_fusion.datasets.tiles import TileDataset
from dataclasses import asdict
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.lit import RegressionLit
from xenium_hne_fusion.train.utils import resolve_training_paths, set_fast_dev_run_settings
from xenium_hne_fusion.models.utils import get_morph_encoder_and_transform, get_expr_encoder_and_transform

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

L.seed_everything(0)
torch.set_float32_matmul_precision('high')

NUM_CELL_TYPES = 39

def train(cfg: Config, debug: bool | None = None):
    debug = debug if debug is not None else cfg.debug
    if debug or cfg.trainer.fast_dev_run:
        cfg = set_fast_dev_run_settings(cfg)
    cfg, output_dir = resolve_training_paths(cfg)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data.panel_path is not None:
        panel_path = cfg.data.panel_path
        assert panel_path.exists(), f"Panel file not found: {panel_path}"
        panel = yaml.safe_load(panel_path.read_text()) or {}
        if cfg.data.source_panel is None:
            cfg.data.source_panel = panel['source_panel']
        if cfg.data.target_panel is None:
            cfg.data.target_panel = panel['target_panel']

    assert cfg.data.source_panel is not None, "cfg.data.source_panel must be set"
    assert cfg.data.target_panel is not None, "cfg.data.target_panel must be set"
    assert set(cfg.data.source_panel).isdisjoint(set(cfg.data.target_panel))

    num_target_genes = len(cfg.data.target_panel) if cfg.data.target_panel else None
    num_source_genes = len(cfg.data.source_panel) if cfg.data.source_panel else None

    # ---------------- Model ----------------
    morph_encoder_name = cfg.backbone.morph_encoder_name
    morph_encoder_kws = cfg.backbone.morph_encoder_kws or {}

    expr_encoder_name = cfg.backbone.expr_encoder_name
    expr_encoder_cfg = cfg.backbone.expr_encoder_kws or {}

    assert morph_encoder_name is not None or expr_encoder_name is not None, "At least one encoder must be specified"

    # VISION
    morph_encoder, image_transform, morph_encoder_dim = get_morph_encoder_and_transform(
        morph_encoder_name=morph_encoder_name, **morph_encoder_kws)

    # EXPR
    kws = {**expr_encoder_cfg, 'input_dim': num_source_genes}
    expr_encoder, expr_transform, expr_encoder_dim = get_expr_encoder_and_transform(
        expr_encoder_name=expr_encoder_name, source_panel=cfg.data.source_panel, **kws)

    backbone = FusionModel(
        # runtime resolved encoders
        expr_encoder=expr_encoder,
        morph_encoder=morph_encoder,
        expr_encoder_dim=expr_encoder_dim,
        morph_encoder_dim=morph_encoder_dim,

        # FusionModel kwargs from config
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

    if cfg.backbone.fusion_stage == 'late' and cfg.backbone.fusion_strategy == 'concat':
        embed_dim = morph_encoder_dim * 2
    else:
        embed_dim = morph_encoder_dim or expr_encoder_dim

    assert cfg.head.num_hidden_layers == 0, 'Use only linear probing head for now...'

    head = Head(
        input_dim=embed_dim,
        output_dim=NUM_CELL_TYPES,
        hidden_dim=cfg.head.hidden_dim,
        num_hidden_layers=cfg.head.num_hidden_layers,
        dropout=cfg.head.dropout,
    )

    lit = RegressionLit(
        backbone=backbone,
        head=head,
        num_outputs=NUM_CELL_TYPES,
        batch_key="modalities",
        target_key=cfg.lit.target_key,
        lr_head=cfg.lit.lr_head,
        lr_backbone=cfg.lit.lr_backbone,
        weight_decay=cfg.lit.weight_decay,
        eta=cfg.lit.eta,
        schedule=cfg.lit.schedule,
        num_warmup_epochs=cfg.lit.num_warmup_epochs,
        save_hparams=False,  # everything is logged to wandb by the config
    )

    # ---------------- Data ----------------
    ds_fit = TileDataset(
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        panel=cfg.data.source_panel,
        include_image=morph_encoder is not None,
        include_expr=expr_encoder is not None,
        image_transform=image_transform,
        expr_transform=expr_transform,
        expr_pool=cfg.data.expr_pool,
        split="fit",
        drop_nan_columns=True,
        id_key="id",
        cache_dir=cfg.data.cache_dir,
    )
    ds_fit.setup()

    ds_val = TileDataset(
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        panel=cfg.data.source_panel,
        include_image=morph_encoder is not None,
        include_expr=expr_encoder is not None,
        image_transform=image_transform,
        expr_transform=expr_transform,
        expr_pool=cfg.data.expr_pool,
        split="val",
        drop_nan_columns=True,
        id_key="id",
        cache_dir=cfg.data.cache_dir,
    )
    ds_val.setup()

    ds_test = TileDataset(
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        panel=cfg.data.source_panel,
        include_image=morph_encoder is not None,
        include_expr=expr_encoder is not None,
        image_transform=image_transform,
        expr_transform=expr_transform,
        expr_pool=cfg.data.expr_pool,
        split="test",
        drop_nan_columns=True,
        id_key="id",
        cache_dir=cfg.data.cache_dir,
    )
    ds_test.setup()

    global_batch_size = cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
    dataloader_kws = dict(
        batch_size=cfg.data.batch_size,
        pin_memory=cfg.data.num_workers > 0,
        persistent_workers=cfg.data.num_workers > 0,
        num_workers=cfg.data.num_workers,
    )
    if cfg.data.num_workers > 0 and cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = cfg.data.prefetch_factor

    dl_fit = DataLoader(
        ds_fit,
        shuffle=True,
        **dataloader_kws,
    )

    dl_val = DataLoader(
        ds_val,
        **dataloader_kws,
    )

    dl_test = DataLoader(
        ds_test,
        **dataloader_kws,
    )

    log_every_n_steps = min(50, len(dl_fit))

    # ---------------- Logging ----------------
    wb_logger = WandbLogger(
        entity="chuv",
        save_dir=logs_dir,
        **asdict(cfg.wandb),
        config={
            # runtime / resolved paths
            "global_batch_size": global_batch_size,
            "num_target_genes": num_target_genes,
            "num_source_genes": num_source_genes,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            # resolved config
            **asdict(cfg),
        },
    )

    # ---------------- Callbacks ----------------
    monitor = "val/mse_mean"
    mode = "min"

    callbacks = [
        LogModelStats(),
        LogWandbRunMetadataCallback(),
        LogCheckpointPathsCallback(),
        ModelCheckpoint(monitor=monitor, mode=mode, filename='best-{epoch}-{step}', save_last=False),
        # save the best model, we cannot inject the metric due to the / in name
        ModelCheckpoint(monitor=None, save_last="link"),  # save the latest model for resuming
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor=monitor, mode=mode, patience=10),
        TestCache(exclude_keys=['modalities.image', 'loss', 'z'])  # exclude data from cache
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

    return dict(head=head,
                backbone=backbone,
                lit=lit,
                trainer=trainer,
                ds_fit=ds_fit,
                ds_val=ds_val)

if __name__ == "__main__":
    from pathlib import Path
    from jsonargparse import auto_cli

    def main(cfg: Path, debug: bool = False) -> None:
        """Train cell type prediction model.

        Args:
            cfg: Path to training config YAML.
            debug: Enable fast-dev-run debug mode (2 batches, batch_size=2).
        """
        train(Config.from_yaml(cfg), debug=debug)

    auto_cli(main, as_positional=False)
