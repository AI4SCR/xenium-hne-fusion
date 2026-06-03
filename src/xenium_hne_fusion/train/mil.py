from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from ai4bmr_learn.callbacks.log_model_checkpoint_paths import LogCheckpointPathsCallback
from ai4bmr_learn.callbacks.log_model_stats import LogModelStats
from ai4bmr_learn.callbacks.log_wandb_run_metadata import LogWandbRunMetadataCallback
from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.datasets import BagsDataset, pad_bags_collate
from ai4bmr_learn.lit.mil import ClassificationMILLit, RegressionMILLit, SurvivalMILLit
from ai4bmr_learn.models.mil import (
    AttentionAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    SimpleAttentionAggregation,
    TransformerAttentionAggregation,
)
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import DataLoader

from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.train.config import Config as SupervisedConfig
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.train.supervised import set_fast_dev_run_settings
from xenium_hne_fusion.utils.getters import get_managed_paths


@dataclass
class ResolvedPretrainedRun:
    checkpoint_path: Path
    source_config: SupervisedConfig


class MILBagsDataset(BagsDataset):
    name = "MILBags"

    def __init__(self, *args, embedding_key: str = "z", **kwargs):
        kwargs.setdefault("id_key", "sample_id")
        super().__init__(*args, bag_id_key="sample_id", **kwargs)
        self.embedding_key = embedding_key

    def __getitem__(self, idx) -> dict[str, Any]:
        assert self.bag_ids is not None and self.bag_items is not None, "setup"
        bag_id = self.bag_ids[idx]
        items = self.bag_items[bag_id]
        assert len(items) == 1, f"expected 1 item per bag, got {len(items)}"
        bag_payload = torch.load(Path(items[0]["path"]), map_location="cpu")
        assert (
            isinstance(bag_payload, dict) and self.embedding_key in bag_payload
        ), "bag_payload"
        embeddings = torch.as_tensor(
            bag_payload[self.embedding_key], dtype=torch.float32
        )
        assert embeddings.ndim == 2, "embeddings_ndim"
        item: dict[str, Any] = {"bag_id": bag_id, "bag": embeddings}
        if self.metadata is not None:
            item["metadata"] = self.metadata.loc[bag_id].to_dict()
        if self.transform is not None:
            item = self.transform(item)
        return item


def resolve_pretrained_run(pretrained_cfg) -> ResolvedPretrainedRun:
    api = wandb.Api()
    run = api.run(
        f"{pretrained_cfg.entity}/{pretrained_cfg.project}/{pretrained_cfg.run_id}"
    )
    raw_config = dict(run.config)
    checkpoint_path_value = raw_config.get("best_model_path")
    assert checkpoint_path_value is not None, "best_model_path"
    checkpoint_path = Path(os.path.expandvars(checkpoint_path_value)).expanduser()
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    return ResolvedPretrainedRun(
        checkpoint_path=checkpoint_path,
        source_config=SupervisedConfig.from_dict(raw_config),
    )


def build_mil_metadata(source_cfg: SupervisedConfig, run_dir: Path) -> Path:
    import pandas as pd
    from xenium_hne_fusion.train.utils import resolve_training_paths

    output_path = run_dir / "metadata.parquet"
    if output_path.exists():
        logger.info(f"Reusing existing MIL metadata -> {output_path}")
        return output_path

    src_cfg, _ = resolve_training_paths(source_cfg)
    assert src_cfg.data.metadata_path is not None and src_cfg.data.metadata_path.exists(), \
        f"supervised split parquet not found: {src_cfg.data.metadata_path}"

    # Assert each sample belongs to exactly one split before deduplicating
    split_df = pd.read_parquet(src_cfg.data.metadata_path, columns=["sample_id", "split"])
    splits_per_sample = split_df.groupby("sample_id")["split"].nunique()
    assert (splits_per_sample == 1).all(), \
        f"samples belong to multiple splits: {splits_per_sample[splits_per_sample > 1].index.tolist()}"
    sample_splits = split_df.drop_duplicates("sample_id").set_index("sample_id")["split"]

    # Join with structured (clinical) metadata
    managed = get_managed_paths(src_cfg.data.name)
    structured_path = managed.structured_dir / "metadata.parquet"
    assert structured_path.exists(), f"structured metadata not found: {structured_path}"

    metadata = pd.read_parquet(structured_path)
    if "sample_id" not in metadata.columns:
        metadata = metadata.reset_index()
    assert "sample_id" in metadata.columns, "sample_id missing from structured metadata"

    merged = sample_splits.reset_index().merge(metadata, on="sample_id", how="left")
    merged = merged.set_index("sample_id")
    assert not merged.index.duplicated().any(), "duplicate sample_ids after merge"

    # Drop columns with any NaN
    valid_cols = [col for col in merged.columns if not merged[col].isna().any()]
    merged = merged[valid_cols]

    logger.warning(
        "MIL metadata uses supervised splits — not stratified on MIL target. "
        "Set metadata_path in config to provide custom stratified splits."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path)
    logger.info(f"Saved MIL metadata -> {output_path} {merged.shape}, columns: {merged.columns.tolist()}")
    return output_path


def resolve_mil_paths(cfg: MILConfig) -> tuple[MILConfig, Path]:
    assert cfg.data.name is not None, "cfg.data.name"
    managed = get_managed_paths(cfg.data.name)
    output_dir = managed.output_dir

    run_dir = output_dir / "mil" / cfg.pretrained.run_id

    cfg.data.cache_dir = _resolve_path(
        cfg.data.cache_dir, root=run_dir / "cache", default=run_dir / "cache"
    )

    return cfg, run_dir


def build_aggregator(cfg: MILConfig, input_dim: int):
    name = cfg.aggregator.name
    match name:
        case "mean":
            return MeanAggregation(input_dim=input_dim)
        case "max":
            return MaxAggregation(input_dim=input_dim)
        case "min":
            return MinAggregation(input_dim=input_dim)
        case "simple_attention":
            return SimpleAttentionAggregation(input_dim=input_dim)
        case "attention":
            return AttentionAggregation(
                input_dim=input_dim,
                hidden_dim=cfg.aggregator.hidden_dim,
                gated=cfg.aggregator.gated,
            )
        case "transformer_attention":
            return TransformerAttentionAggregation(
                input_dim=input_dim,
                hidden_dim=cfg.aggregator.hidden_dim,
                dropout=cfg.aggregator.dropout,
                num_heads=cfg.aggregator.num_heads,
            )
        case _:
            raise ValueError(f"Unknown aggregator: {name}")


def build_mil_module(*, cfg: MILConfig, input_dim: int, num_classes: int | None = None):
    aggregator = build_aggregator(cfg, input_dim=input_dim)
    output_dim = int(num_classes or 0) if cfg.task.kind == "classification" else 1
    assert output_dim > 0, "output_dim"
    head = Head(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=cfg.head.hidden_dim,
        num_hidden_layers=cfg.head.num_hidden_layers,
        dropout=cfg.head.dropout,
    )
    common_kws = dict(
        aggregator=aggregator,
        head=head,
        lr_head=cfg.lit.lr_head,
        lr_aggregator=cfg.lit.lr_aggregator,
        weight_decay=cfg.lit.weight_decay,
        eta=cfg.lit.eta,
        schedule=cfg.lit.schedule,
        max_epochs=cfg.trainer.max_epochs,
        num_warmup_epochs=cfg.lit.num_warmup_epochs,
        metric_names=cfg.lit.metric_names,
    )
    match cfg.task.kind:
        case "classification":
            assert num_classes is not None, "num_classes"
            return ClassificationMILLit(
                num_classes=num_classes,
                target_key=cfg.lit.target_key,
                **common_kws,
            )
        case "regression":
            return RegressionMILLit(
                num_outputs=1,
                loss=cfg.lit.loss,
                target_key=cfg.lit.target_key,
                **common_kws,
            )
        case "survival":
            return SurvivalMILLit(
                time_key=cfg.lit.time_key,
                event_key=cfg.lit.event_key,
                **common_kws,
            )
        case _:
            raise ValueError(f"Received unsupported `cfg.task.kind` {cfg.task.kind}")


def train(cfg: MILConfig, config_path: str | None = None):
    L.seed_everything(0)

    if cfg.debug or cfg.trainer.fast_dev_run:
        cfg = set_fast_dev_run_settings(cfg)
        cfg.data.batch_size = 2
        cfg.data.num_workers = 0
        cfg.data.prefetch_factor = None

    resolved_run = resolve_pretrained_run(cfg.pretrained)
    cfg, run_dir = resolve_mil_paths(cfg)
    if cfg.wandb.name is None:
        cfg.wandb.name = resolved_run.source_config.wandb.name

    cache_subdir = "debug" if cfg.debug else "predictions"
    cfg.data.items_path = run_dir / cache_subdir / "bags.json"
    assert cfg.data.items_path.exists(), (
        f"bags.json not found at {cfg.data.items_path}. "
        "Run scripts/artifacts/cache_predictions.py first."
    )
    if cfg.data.metadata_path is None:
        cfg.data.metadata_path = build_mil_metadata(resolved_run.source_config, run_dir)
    else:
        assert cfg.data.metadata_path.exists(), f"metadata_path not found: {cfg.data.metadata_path}"
        logger.info(f"Using custom MIL metadata -> {cfg.data.metadata_path}")

    dataset_kws = dict(
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
    )
    ds_fit = MILBagsDataset(**dataset_kws, split=Split.FIT.value)
    ds_fit.setup()
    ds_val = MILBagsDataset(**dataset_kws, split=Split.VAL.value)
    ds_val.setup()
    ds_test = MILBagsDataset(**dataset_kws, split=Split.TEST.value)
    ds_test.setup()

    example_bag = ds_fit[0]["bag"]
    input_dim = int(example_bag.shape[1])
    num_classes = cfg.task.num_classes
    if cfg.task.kind == "classification":
        assert (
            num_classes is not None and num_classes > 1
        ), "task.num_classes must be set for classification"
    mil_lit = build_mil_module(cfg=cfg, input_dim=input_dim, num_classes=num_classes)

    dataloader_kws = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=pad_bags_collate,
        pin_memory=cfg.data.num_workers > 0,
        persistent_workers=cfg.data.num_workers > 0,
    )
    if cfg.data.num_workers > 0 and cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = cfg.data.prefetch_factor

    dl_fit = DataLoader(ds_fit, shuffle=True, **dataloader_kws)
    # full batch evaluation for val and test to avoid no-event batches and to get better loss estimate
    dataloader_kws.pop("batch_size")
    dl_val = DataLoader(ds_val, shuffle=False, **dataloader_kws, batch_size=len(ds_val))
    dl_test = DataLoader(
        ds_test, shuffle=False, **dataloader_kws, batch_size=len(ds_test)
    )

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    wb_logger = WandbLogger(
        entity=cfg.pretrained.entity,
        save_dir=logs_dir,
        **asdict(cfg.wandb),
        config={
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "config_path": config_path,
            **asdict(cfg),
        },
    )
    callbacks = [
        LogModelStats(),
        LogWandbRunMetadataCallback(),
        LogCheckpointPathsCallback(),
        LearningRateMonitor(logging_interval="epoch"),
        # EarlyStopping(monitor="loss/val", mode="min", patience=15),
    ]
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=wb_logger,
        callbacks=callbacks,
        default_root_dir=run_dir,
        **asdict(cfg.trainer),
    )
    trainer.fit(model=mil_lit, train_dataloaders=dl_fit, val_dataloaders=dl_val)
    if not cfg.trainer.fast_dev_run:
        trainer.test(model=mil_lit, dataloaders=dl_test)
    wandb.finish()

    return {
        "resolved_run": resolved_run,
        "trainer": trainer,
        "lit": mil_lit,
        "ds_fit": ds_fit,
        "ds_val": ds_val,
        "ds_test": ds_test,
    }


def main(cfg: MILConfig, config_path: str | None = None) -> None:
    train(cfg, config_path=config_path)


def _resolve_path(
    path: Path | None, *, root: Path | None = None, default: Path | None = None
) -> Path | None:
    if path is None:
        return default
    path = Path(os.path.expandvars(path))
    if path.is_absolute():
        return path
    assert root is not None, "root"
    return root / path
