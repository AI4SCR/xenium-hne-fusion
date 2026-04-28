from __future__ import annotations

import os
from dataclasses import asdict, dataclass
import tempfile
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import wandb
from ai4bmr_learn.callbacks.cache import PredictionCache
from ai4bmr_learn.datasets import pad_bags_collate, write_mil_items_from_cache
from ai4bmr_learn.datasets.items import Items
from ai4bmr_learn.datasets.utils import filter_items_and_metadata
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
import torch
from torch.utils.data import DataLoader

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.train.config import Config as SupervisedConfig
from xenium_hne_fusion.train.mil_config import MILConfig
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit, set_fast_dev_run_settings
from xenium_hne_fusion.train.utils import prepare_training_config
from xenium_hne_fusion.utils.getters import get_managed_paths


@dataclass
class ResolvedPretrainedRun:
    checkpoint_path: Path
    source_config: SupervisedConfig
    raw_config: dict[str, Any]


class MetadataBagsDataset(Items):
    name = "MetadataBags"

    def __init__(
        self,
        *args,
        task_kind: str,
        target_key: str = "target",
        embedding_key: str = "z",
        bag_id_key: str = "sample_id",
        **kwargs,
    ):
        kwargs.setdefault("id_key", bag_id_key)
        super().__init__(*args, **kwargs)
        assert task_kind in {"regression", "classification"}, f"Unknown task_kind: {task_kind}"
        self.task_kind = task_kind
        self.target_key = target_key
        self.embedding_key = embedding_key
        self.bag_id_key = bag_id_key
        self.embedding_path_key = f"{embedding_key}_path"
        self.items_by_bag_id: dict[str, dict[str, Any]] | None = None
        self.bag_ids: list[str] | None = None

    def setup(self) -> None:
        super().setup()
        assert self.items is not None, "items"

        if self.metadata_path is not None:
            assert self.id_key is not None, "id_key"
            item_ids = [item[self.id_key] for item in self.items]
            metadata = pd.read_parquet(self.metadata_path)
            self.item_ids, self.metadata = filter_items_and_metadata(
                item_ids=item_ids,
                metadata=metadata,
                split=self.split,
                drop_nan_columns=self.drop_nan_columns,
            )
            item_id_set = set(self.item_ids)
            self.items = [item for item in self.items if item[self.id_key] in item_id_set]

        items_by_bag_id: dict[str, dict[str, Any]] = {}
        for item in self.items:
            assert self.bag_id_key in item, "bag_id"
            assert self.embedding_path_key in item, "embedding_path"
            bag_id = str(item[self.bag_id_key])
            assert bag_id not in items_by_bag_id, "duplicate_bag_id"
            items_by_bag_id[bag_id] = item

        self.items_by_bag_id = items_by_bag_id
        self.bag_ids = list(items_by_bag_id)
        assert self.bag_ids, "bags"

        if self.metadata is not None:
            metadata = self.metadata.copy()
            metadata.index = metadata.index.astype(str)
            metadata = metadata.loc[~metadata.index.duplicated(keep="first")]
            missing_bags = set(self.bag_ids) - set(metadata.index)
            assert not missing_bags, "metadata_bags"
            self.metadata = metadata.loc[self.bag_ids]

    def __len__(self) -> int:
        assert self.bag_ids is not None, "setup"
        return len(self.bag_ids)

    def get_bag_id(self, idx: int) -> str:
        assert self.bag_ids is not None, "setup"
        return self.bag_ids[idx]

    def get_bag_item(self, idx: int) -> dict[str, Any]:
        assert self.items_by_bag_id is not None, "setup"
        return self.items_by_bag_id[self.get_bag_id(idx)]

    def get_bag(self, idx: int) -> torch.Tensor:
        bag_item = self.get_bag_item(idx)
        bag_payload = torch.load(Path(bag_item[self.embedding_path_key]), map_location="cpu")
        assert isinstance(bag_payload, dict), "bag_payload"
        assert self.embedding_key in bag_payload, "embedding"
        embeddings = torch.as_tensor(bag_payload[self.embedding_key], dtype=torch.float32)
        assert embeddings.ndim == 2, "embeddings_ndim"
        return embeddings

    def get_metadata(self, idx: int) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        return self.metadata.loc[self.get_bag_id(idx)].to_dict()

    def __getitem__(self, idx) -> dict[str, Any]:
        item = {
            "bag_id": self.get_bag_id(idx),
            "bag": self.get_bag(idx),
        }
        metadata = self.get_metadata(idx)
        assert metadata is not None, "metadata"
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
    metadata_path: Path,
    target_key: str,
    task_kind: str,
    output_path: Path,
) -> Path:
    assert target_key.startswith("metadata."), target_key
    target_column = target_key.removeprefix("metadata.")
    assert target_column, "target_column"

    metadata = pd.read_parquet(metadata_path)
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
        sample_metadata[target_column] = categorical.codes.astype("int64")
    else:
        numeric = pd.to_numeric(sample_metadata[target_column], errors="raise")
        sample_metadata[target_column] = numeric.astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_metadata.to_parquet(output_path)
    logger.info(f"Saved MIL sample metadata -> {output_path}")
    return output_path


def resolve_mil_paths(cfg: MILConfig) -> tuple[MILConfig, Path]:
    assert cfg.data.name is not None, "cfg.data.name"
    assert cfg.data.metadata_path is not None, "cfg.data.metadata_path"
    output_dir = get_managed_paths(cfg.data.name).output_dir
    run_root = output_dir / "mil" / (cfg.wandb.name or cfg.pretrained.run_id)
    cfg.data.metadata_path = _resolve_path(cfg.data.metadata_path, root=output_dir / "splits")
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
    resolved = prepare_training_config(resolved_run.source_config)
    lit = build_supervised_lit(resolved.cfg, checkpoint_path=resolved_run.checkpoint_path)
    lit.eval()

    # The pretrained supervised run defines the tile item universe used to build
    # embeddings for the downstream MIL stage.
    dataset_kws = build_supervised_dataset_kws(resolved.cfg)
    ds = TileDataset(**dataset_kws, split=None)
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
    cache_files = sorted(prediction_dir.glob("*.pt")) if prediction_dir.exists() else []
    if not cache_files:
        prediction_cache = PredictionCache(
            save_dir=cfg.data.cache_dir,
            include_keys=["z", "id", "sample_id"],
            save_in_batches=True,
        )
        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            logger=False,
            callbacks=[prediction_cache],
            default_root_dir=run_root,
        )
        trainer.predict(model=lit, dataloaders=dl)

    # MIL training consumes the cached bag manifest plus one `<bag_id>.pt` file
    # per bag written by the shared bag writer.
    items_path = run_root / "bags.json"
    write_mil_items_from_prediction_cache(cache_dir=prediction_dir, items_path=items_path)
    return items_path


def write_mil_items_from_prediction_cache(*, cache_dir: Path, items_path: Path) -> Path:
    cache_dir = Path(cache_dir)
    items_path = Path(items_path)
    assert items_path.name == "bags.json", items_path.name
    cache_files = sorted(cache_dir.glob("*.pt"))
    assert cache_files, "cache_files"

    # `write_mil_items_from_cache` groups bags by the values found at `id_key`.
    # The prediction cache keeps both tile `id` and bag-level `sample_id`, so we
    # rewrite only the temporary writer input to group by sample while still
    # calling the shared helper with `id_key="id"`. The resulting manifest is
    # the `bags.json` consumed by MIL training.
    with tempfile.TemporaryDirectory(prefix="mil-cache-", dir=items_path.parent) as tmp_dir:
        tmp_cache_dir = Path(tmp_dir)
        for cache_file in cache_files:
            outputs = torch.load(cache_file, map_location="cpu")
            rewritten_outputs = []
            for output in outputs:
                rewritten_outputs.append(
                    {
                        "id": output["sample_id"],
                        "sample_id": output["sample_id"],
                        "z": output["z"],
                    }
                )
            torch.save(rewritten_outputs, tmp_cache_dir / cache_file.name)
        written_items_path = write_mil_items_from_cache(
            cache_dir=tmp_cache_dir,
            output_dir=items_path.parent,
            id_key="id",
            embedding_key="z",
            bag_id_key="sample_id",
        )
        assert written_items_path == items_path, written_items_path
        return written_items_path


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
        metadata_path=cfg.data.metadata_path,
        target_key=cfg.lit.target_key,
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
        target_column = cfg.lit.target_key.removeprefix("metadata.")
        fit_targets = ds_fit.metadata.loc[ds_fit.bag_ids, target_column]
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
