from pathlib import Path
from typing import Literal

from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import (
    ArtifactsConfig,
    DataConfig,
    FilterConfig,
    ItemsConfig,
    ItemsThresholdConfig,
    PanelConfig,
    SplitConfig,
    TilesConfig,
)


def build_data_parser(*, include_executor: bool = True) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--name", type=str, required=True)
    parser.add_class_arguments(TilesConfig, nested_key="tiles")
    parser.add_class_arguments(FilterConfig, nested_key="filter")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--stage", type=Literal["all", "samples", "finalize"], default="all")
    if include_executor:
        parser.add_argument("--executor", type=Literal["serial", "ray"], default="serial")
    return parser


def namespace_to_data_config(ns) -> DataConfig:
    data = ns.as_dict()
    assert data["tiles"].get("img_size") is not None, "tiles.img_size is required"
    return DataConfig(
        name=data["name"],
        tiles=TilesConfig(**data["tiles"]),
        filter=FilterConfig(**data["filter"]),
    )


def parse_data_args(
    argv: list[str] | None = None,
    *,
    include_executor: bool = True,
) -> tuple[DataConfig, bool, str | None, str]:
    parser = build_data_parser(include_executor=include_executor)
    ns = parser.parse_args(argv)
    executor = ns.executor if include_executor else None
    return namespace_to_data_config(ns), ns.overwrite, executor, ns.stage


def build_artifacts_parser(*, include_overwrite: bool = True) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--name", type=str, required=True)
    parser.add_class_arguments(ItemsConfig, nested_key="items")
    parser.add_class_arguments(SplitConfig, nested_key="split")
    parser.add_class_arguments(PanelConfig, nested_key="panel")
    if include_overwrite:
        parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def namespace_to_artifacts_config(ns) -> ArtifactsConfig:
    data = ns.as_dict()
    items = _drop_internal_keys(data["items"])
    items_filter = _drop_internal_keys(items["filter"])
    split = _drop_internal_keys(data["split"])
    panel = data.get("panel")
    if panel is not None:
        panel = _drop_internal_keys(panel)
        if panel.get("metadata_path") is not None:
            panel["metadata_path"] = Path(panel["metadata_path"])
        if not any(value is not None for value in panel.values()):
            panel = None
    return ArtifactsConfig(
        name=data["name"],
        items=ItemsConfig(
            name=items["name"],
            filter=ItemsThresholdConfig(**items_filter),
        ),
        split=SplitConfig(**split),
        panel=None if panel is None else PanelConfig(**panel),
    )


def parse_artifacts_args(argv: list[str] | None = None) -> tuple[ArtifactsConfig, bool]:
    ns = build_artifacts_parser().parse_args(argv)
    return namespace_to_artifacts_config(ns), ns.overwrite


def _drop_internal_keys(data: dict) -> dict:
    return {key: value for key, value in data.items() if not key.startswith("__")}


build_processing_parser = build_data_parser
namespace_to_processing_config = namespace_to_data_config
parse_processing_args = parse_data_args
