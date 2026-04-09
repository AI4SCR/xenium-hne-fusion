from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import ArgumentParser

from xenium_hne_fusion.train.config import (
    BackboneConfig,
    Config,
    DataLoaderConfig,
    HeadConfig,
    LitConfig,
    TaskConfig,
    TrainerConfig,
    WandbConfig,
)


def build_train_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_class_arguments(TaskConfig, nested_key="task")
    parser.add_class_arguments(HeadConfig, nested_key="head")
    parser.add_class_arguments(BackboneConfig, nested_key="backbone")
    parser.add_class_arguments(DataLoaderConfig, nested_key="data")
    parser.add_class_arguments(LitConfig, nested_key="lit")
    parser.add_class_arguments(TrainerConfig, nested_key="trainer")
    parser.add_class_arguments(WandbConfig, nested_key="wandb")
    return parser


def namespace_to_config(ns: Any) -> Config:
    data = ns.as_dict()
    return Config(
        debug=data["debug"],
        task=TaskConfig(**data["task"]),
        head=HeadConfig(**data["head"]),
        backbone=BackboneConfig(**data["backbone"]),
        data=DataLoaderConfig(**data["data"]),
        lit=LitConfig(**data["lit"]),
        trainer=TrainerConfig(**data["trainer"]),
        wandb=WandbConfig(**data["wandb"]),
    )


def serialize_cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False, default_flow_style=True).strip()
    return str(value)


def parse_training_config(config_path: Path, overrides: dict[str, Any] | None = None) -> Config:
    args = ["--config", str(config_path)]
    for key, value in (overrides or {}).items():
        args.extend([f"--{key}", serialize_cli_value(value)])
    namespace = build_train_parser().parse_args(args)
    return namespace_to_config(namespace)


def config_to_document(cfg: Config) -> dict[str, Any]:
    return _to_yaml_primitive(asdict(cfg))


def _to_yaml_primitive(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_yaml_primitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yaml_primitive(item) for item in value]
    return value
