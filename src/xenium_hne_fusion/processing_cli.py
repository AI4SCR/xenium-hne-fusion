from pathlib import Path
from typing import Literal

from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import (
    FilterConfig,
    ItemsConfig,
    ItemsThresholdConfig,
    PanelConfig,
    ProcessingConfig,
    SplitConfig,
    TilesConfig,
)


def build_processing_parser(*, include_executor: bool = True) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--name", type=str, required=True)
    parser.add_class_arguments(TilesConfig, nested_key="tiles")
    parser.add_class_arguments(FilterConfig, nested_key="filter")
    parser.add_class_arguments(ItemsConfig, nested_key="items")
    parser.add_class_arguments(SplitConfig, nested_key="split")
    parser.add_class_arguments(PanelConfig, nested_key="panel")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--stage", type=Literal["all", "samples", "finalize"], default="all")
    if include_executor:
        parser.add_argument("--executor", type=Literal["serial", "ray"], default="serial")
    return parser


def namespace_to_processing_config(ns) -> ProcessingConfig:
    data = ns.as_dict()
    items = _drop_internal_keys(data["items"])
    items_filter = _drop_internal_keys(items["filter"])
    split = _drop_internal_keys(data["split"])
    panel = _drop_internal_keys(data["panel"]) if data.get("panel") else None
    assert data["tiles"].get("img_size") is not None, "tiles.img_size is required"
    return ProcessingConfig(
        name=data["name"],
        tiles=TilesConfig(**data["tiles"]),
        filter=FilterConfig(**data["filter"]),
        items=ItemsConfig(
            name=items["name"],
            filter=ItemsThresholdConfig(**items_filter),
        ),
        split=SplitConfig(**split),
        panel=None if panel is None else PanelConfig(**panel),
    )


def _drop_internal_keys(data: dict) -> dict:
    return {key: value for key, value in data.items() if not key.startswith("__")}


def parse_processing_args(
    argv: list[str] | None = None,
    *,
    include_executor: bool = True,
) -> tuple[ProcessingConfig, bool, str | None, str]:
    parser = build_processing_parser(include_executor=include_executor)
    ns = parser.parse_args(argv)
    executor = ns.executor if include_executor else None
    return namespace_to_processing_config(ns), ns.overwrite, executor, ns.stage
