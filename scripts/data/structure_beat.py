"""Structure raw BEAT data into 01_structured canonical layout."""

import sys

from dotenv import load_dotenv
from loguru import logger

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.structure import structure_metadata, structure_sample
from xenium_hne_fusion.utils.getters import build_pipeline_config


def main(data_cfg: DataConfig) -> None:
    load_dotenv()
    assert data_cfg.name == "beat", f"Expected dataset='beat', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)

    metadata_path = cfg.raw_dir / "metadata.parquet"
    if metadata_path.exists():
        structure_metadata(metadata_path, cfg.paths.structured_dir)

    sample_dirs = sorted(p for p in cfg.raw_dir.iterdir() if p.is_dir())
    logger.info(f"Found {len(sample_dirs)} samples in {cfg.raw_dir}")
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        wsi_path = sample_dir / "region.tiff"
        tx_path = sample_dir / "transcripts" / "transcripts.parquet"
        cells_path = sample_dir / "cells.parquet"
        structure_sample(
            sample_id,
            wsi_path,
            tx_path,
            cfg.paths.structured_dir,
            cells_path=cells_path if cells_path.exists() else None,
        )


def cli(argv: list[str] | None = None) -> int:
    data_cfg, _, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
