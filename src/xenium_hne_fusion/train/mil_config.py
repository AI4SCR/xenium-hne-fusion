from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from xenium_hne_fusion.train.config import HeadConfig, TrainerConfig, WandbConfig, _merge_dataclass


@dataclass
class PretrainedConfig:
    entity: str = "chuv"
    project: str = ""
    run_id: str = ""


@dataclass
class MILDataConfig:
    name: str | None = None
    metadata_path: Path | None = None
    cache_dir: Path | None = None
    batch_size: int = 8
    num_workers: int = 10
    prefetch_factor: int | None = 4


@dataclass
class MILTaskConfig:
    kind: Literal["regression", "classification"] = "regression"


@dataclass
class AggregatorConfig:
    name: Literal["mean", "max", "min", "simple_attention", "attention"] = "attention"
    hidden_dim: int = 128
    gated: bool = False


@dataclass
class MILLitConfig:
    target_key: str = "target"
    lr_head: float = 1e-3
    lr_aggregator: float = 1e-4
    weight_decay: float = 1e-2
    eta: float = 0.0
    schedule: Literal["cosine"] | None = None
    num_warmup_epochs: int = 10
    loss: Literal["mse", "huber"] = "mse"
    metric_names: list[str] | None = None


@dataclass
class MILConfig:
    debug: bool = False
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    data: MILDataConfig = field(default_factory=MILDataConfig)
    task: MILTaskConfig = field(default_factory=MILTaskConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    lit: MILLitConfig = field(default_factory=MILLitConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "MILConfig":
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        return _merge_dataclass(cls, data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MILConfig":
        return _merge_dataclass(cls, data)
