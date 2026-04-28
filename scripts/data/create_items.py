"""Build items JSON for TileDataset from processed tile data."""

import sys

from dotenv import load_dotenv

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.pipeline import create_items as create_items_stage
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.utils.getters import build_pipeline_config


def main(data_cfg: DataConfig, overwrite: bool = False) -> None:
    load_dotenv()
    cfg = build_pipeline_config(data_cfg)
    create_items_stage(cfg, kernel_size=cfg.data.tiles.kernel_size, overwrite=overwrite)


def cli(argv: list[str] | None = None) -> int:
    data_cfg, overwrite_arg, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg, overwrite=overwrite_arg)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
