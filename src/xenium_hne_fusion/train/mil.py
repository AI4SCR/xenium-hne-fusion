from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
import wandb
from ai4bmr_learn.callbacks.cache import PredictionCache
from ai4bmr_learn.datasets import BagsDataset, pad_bags_collate, write_mil_items_from_cache
from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.lit.mil import ClassificationMILLit, RegressionMILLit
from ai4bmr_learn.models.mil import (
    AttentionAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    SimpleAttentionAggregation,
)
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import DataLoader

from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.train.config import Config as SupervisedConfig
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.train.supervised import build_supervised_lit, set_fast_dev_run_settings
from xenium_hne_fusion.utils.getters import get_managed_paths


@dataclass
class ResolvedPretrainedRun:
    checkpoint_path: Path
    source_config: SupervisedConfig
    raw_config: dict[str, Any]


class MetadataBagsDataset(BagsDataset):
    name = "MetadataBags"

    def __init__(self, *args, task_kind: str, target_key: str = "target", **kwargs):
        super().__init__(*args, **kwargs)
        assert task_kind in {"regression", "classification"}, f"Unknown task_kind: {task_kind}"
        self.task_kind = task_kind
        self.target_key = target_key

    def __getitem__(self, idx) -> dict[str, Any]:
        item = {
            "bag_id": self.get_bag_id(idx),
            "bag": self.get_bag(idx),
        }
        metadata = self.get_metadata(idx)
        assert metadata is not None, "metadata"
        assert self.target_key in metadata, "target"
        target = metadata[self.target_key]
        assert pd.notna(target), "target_nan"
        dtype = torch.long if self.task_kind == "classification" else torch.float32
        item["target"] = torch.as_tensor(target, dtype=dtype)
        item["metadata"] = metadata
        if self.transform is not None:
            item = self.transform(item)
        return item


def resolve_pretrained_run(pretrained_cfg) -> ResolvedPretrainedRun:
    api = wandb.Api()
    run = api.run(f"{pretrained_cfg.entity}/{pretrained_cfg.project}/{pretrained_cfg.run_id}")
    raw_config = dict(run.config)
    checkpoint_path_value = raw_config.get("best_model_path")
    assert checkpoint_path_value is not None, "best_model_path"
    checkpoint_path = Path(os.path.expandvars(checkpoint_path_value)).expanduser()
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    return ResolvedPretrainedRun(
        checkpoint_path=checkpoint_path,
        source_config=SupervisedConfig.from_dict(raw_config),
        raw_config=raw_config,
    )


def build_sample_level_mil_metadata(
    *,
    split_metadata_path: Path,
    target_column: str,
    task_kind: str,
    output_path: Path,
) -> Path:
    metadata = pd.read_parquet(split_metadata_path)
    assert "sample_id" in metadata.columns, "sample_id"
    assert Split.COLUMN_NAME.value in metadata.columns, Split.COLUMN_NAME.value
    assert target_column in metadata.columns, target_column

    excluded = {"tile_id", "tile_dir"}
    kept_columns = [column for column in metadata.columns if column not in excluded]
    metadata = metadata[kept_columns].copy()

    grouped = metadata.groupby("sample_id", sort=False, dropna=False)
    for column in [column for column in metadata.columns if column != "sample_id"]:
        nunique = grouped[column].nunique(dropna=False)
        bad = nunique[nunique > 1]
        assert bad.empty, f"inconsistent {column}: {bad.index.tolist()}"

    sample_metadata = grouped.first().reset_index()
    sample_metadata = sample_metadata.set_index("sample_id", drop=True)
    sample_metadata.index = sample_metadata.index.astype(str)

    if task_kind == "classification":
        labels = sample_metadata[target_column]
        assert labels.notna().all(), "target_nan"
        categories = sorted(labels.astype(str).unique().tolist())
        categorical = pd.Categorical(labels.astype(str), categories=categories)
        assert (categorical.codes >= 0).all(), "target_codes"
        sample_metadata["target"] = categorical.codes.astype("int64")
    else:
        numeric = pd.to_numeric(sample_metadata[target_column], errors="raise")
        sample_metadata["target"] = numeric.astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_metadata.to_parquet(output_path)
    logger.info(f"Saved MIL sample metadata -> {output_path}")
    return output_path


def resolve_mil_paths(cfg: MILConfig) -> tuple[MILConfig, Path]:
    assert cfg.data.name is not None, "cfg.data.name"
    assert cfg.data.items_path is not None, "cfg.data.items_path"
    assert cfg.data.split_metadata_path is not None, "cfg.data.split_metadata_path"
    assert cfg.data.target_column is not None, "cfg.data.target_column"
    output_dir = get_managed_paths(cfg.data.name).output_dir
    run_root = output_dir / "mil" / (cfg.wandb.name or cfg.pretrained.run_id)
    cfg.data.items_path = _resolve_path(cfg.data.items_path, root=output_dir / "items")
    cfg.data.split_metadata_path = _resolve_path(cfg.data.split_metadata_path, root=output_dir / "splits")
    cfg.data.cache_dir = _resolve_path(cfg.data.cache_dir, root=run_root / "cache", default=run_root / "cache")
    return cfg, run_root


def build_aggregator(cfg: MILConfig, input_dim: int):
    name = cfg.aggregator.name
    if name == "mean":
        return MeanAggregation(input_dim=input_dim)
    if name == "max":
        return MaxAggregation(input_dim=input_dim)
    if name == "min":
        return MinAggregation(input_dim=input_dim)
    if name == "simple_attention":
        return SimpleAttentionAggregation(input_dim=input_dim)
    if name == "attention":
        return AttentionAggregation(
            input_dim=input_dim,
            hidden_dim=cfg.aggregator.hidden_dim,
            gated=cfg.aggregator.gated,
        )
    raise ValueError(f"Unknown aggregator: {name}")


def build_mil_module(*, cfg: MILConfig, input_dim: int, num_classes: int | None = None):
    aggregator = build_aggregator(cfg, input_dim=input_dim)
    output_dim = 1 if cfg.task.kind == "regression" else int(num_classes or 0)
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
        target_key=cfg.lit.target_key,
        lr_head=cfg.lit.lr_head,
        lr_aggregator=cfg.lit.lr_aggregator,
        weight_decay=cfg.lit.weight_decay,
        eta=cfg.lit.eta,
        schedule=cfg.lit.schedule,
        max_epochs=cfg.trainer.max_epochs,
        num_warmup_epochs=cfg.lit.num_warmup_epochs,
        metric_names=cfg.lit.metric_names,
    )
    if cfg.task.kind == "classification":
        assert num_classes is not None, "num_classes"
        return ClassificationMILLit(num_classes=num_classes, **common_kws)
    return RegressionMILLit(num_outputs=1, loss=cfg.lit.loss, **common_kws)


def extract_mil_embeddings(
    *,
    resolved_run: ResolvedPretrainedRun,
    cfg: MILConfig,
    run_root: Path,
) -> Path:
    source_cfg = resolved_run.source_config
    lit, dataset_kws = build_supervised_lit(source_cfg)

    checkpoint = torch.load(resolved_run.checkpoint_path, map_location="cpu", weights_only=False)
    lit.load_state_dict(checkpoint["state_dict"], strict=True)
    lit.eval()

    ds = dataset_kws["dataset_cls"](**dataset_kws["dataset_kws"], items_path=cfg.data.items_path, split=None)
    ds.setup()

    dataloader_kws = dict(
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=cfg.data.num_workers > 0,
        persistent_workers=cfg.data.num_workers > 0,
        num_workers=cfg.data.num_workers,
    )
    if cfg.data.num_workers > 0 and cfg.data.prefetch_factor is not None:
        dataloader_kws["prefetch_factor"] = cfg.data.prefetch_factor
    dl = DataLoader(ds, **dataloader_kws)

    prediction_dir = cfg.data.cache_dir / "prediction"
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)

    prediction_cache = PredictionCache(save_dir=cfg.data.cache_dir, save_in_batches=True)
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
        callbacks=[prediction_cache],
        default_root_dir=run_root,
    )
    trainer.predict(model=lit, dataloaders=dl)

    items_path = run_root / "mil-items.json"
    write_mil_items_from_cache(
        cache_dir=prediction_dir,
        items_path=items_path,
        id_key="sample_id",
        embedding_key="z",
        bag_id_key="sample_id",
    )
    return items_path


def train(cfg: MILConfig, debug: bool | None = None):
    debug = cfg.debug if debug is None else debug
    if debug or cfg.trainer.fast_dev_run:
        cfg = set_fast_dev_run_settings(cfg)
        cfg.data.batch_size = 2
        cfg.data.num_workers = 0
        cfg.data.prefetch_factor = None

    cfg, run_root = resolve_mil_paths(cfg)
    resolved_run = resolve_pretrained_run(cfg.pretrained)

    mil_items_path = extract_mil_embeddings(resolved_run=resolved_run, cfg=cfg, run_root=run_root)
    sample_metadata_path = build_sample_level_mil_metadata(
        split_metadata_path=cfg.data.split_metadata_path,
        target_column=cfg.data.target_column,
        task_kind=cfg.task.kind,
        output_path=run_root / "sample-metadata.parquet",
    )

    dataset_kws = dict(
        items_path=mil_items_path,
        metadata_path=sample_metadata_path,
        task_kind=cfg.task.kind,
        target_key=cfg.lit.target_key,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
    )
    ds_fit = MetadataBagsDataset(**dataset_kws, split=Split.FIT.value)
    ds_fit.setup()
    ds_val = MetadataBagsDataset(**dataset_kws, split=Split.VAL.value)
    ds_val.setup()
    ds_test = MetadataBagsDataset(**dataset_kws, split=Split.TEST.value)
    ds_test.setup()

    example_bag = ds_fit[0]["bag"]
    input_dim = int(example_bag.shape[1])
    num_classes = None
    if cfg.task.kind == "classification":
        fit_targets = ds_fit.metadata.loc[ds_fit.bag_ids, cfg.lit.target_key]
        num_classes = int(fit_targets.nunique())
        assert num_classes > 1, "num_classes"
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
    dl_val = DataLoader(ds_val, shuffle=False, **dataloader_kws)
    dl_test = DataLoader(ds_test, shuffle=False, **dataloader_kws)

    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    wb_logger = WandbLogger(
        entity=cfg.pretrained.entity,
        save_dir=logs_dir,
        **asdict(cfg.wandb),
        config=asdict(cfg),
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=wb_logger,
        default_root_dir=run_root,
        **asdict(cfg.trainer),
    )
    trainer.fit(model=mil_lit, train_dataloaders=dl_fit, val_dataloaders=dl_val)
    if not cfg.trainer.fast_dev_run:
        trainer.test(model=mil_lit, dataloaders=dl_test)
    wandb.finish()

    return {
        "resolved_run": resolved_run,
        "mil_items_path": mil_items_path,
        "sample_metadata_path": sample_metadata_path,
        "trainer": trainer,
        "lit": mil_lit,
        "ds_fit": ds_fit,
        "ds_val": ds_val,
        "ds_test": ds_test,
    }


def main(cfg: MILConfig, debug: bool | None = None) -> None:
    train(cfg, debug=debug)


def _resolve_path(path: Path | None, *, root: Path | None = None, default: Path | None = None) -> Path | None:
    if path is None:
        return default
    path = Path(path)
    if path.is_absolute():
        return path
    assert root is not None, "root"
    return root / path
