"""Compute statistics for DATA_DIR/03_output/<name>/items/all.json."""

import sys

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.pipeline import compute_all_items_stats
from xenium_hne_fusion.processing_cli import build_data_parser, namespace_to_data_config
from xenium_hne_fusion.utils.getters import build_pipeline_config


def main(
    data_cfg: DataConfig,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    cfg = build_pipeline_config(data_cfg)
    compute_all_items_stats(cfg, overwrite=overwrite)


def build_parser() -> ArgumentParser:
    return build_data_parser(include_executor=False)


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    main(
        namespace_to_data_config(ns),
        overwrite=ns.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
