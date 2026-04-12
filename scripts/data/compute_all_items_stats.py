"""Compute statistics for DATA_DIR/03_output/<name>/items/all.json."""

import sys

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.pipeline import compute_all_items_stats
from xenium_hne_fusion.processing_cli import build_data_parser, namespace_to_data_config
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL, build_pipeline_config


def main(
    data_cfg: DataConfig,
    cell_type_col: str = DEFAULT_CELL_TYPE_COL,
    overwrite: bool = False,
) -> None:
    load_dotenv()
    cfg = build_pipeline_config(data_cfg)
    compute_all_items_stats(cfg, cell_type_col=cell_type_col, overwrite=overwrite)


def build_parser() -> ArgumentParser:
    parser = build_data_parser(include_executor=False)
    parser.add_argument("--cell-type-col", type=str, default=DEFAULT_CELL_TYPE_COL)
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    main(
        namespace_to_data_config(ns),
        cell_type_col=ns.cell_type_col,
        overwrite=ns.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
