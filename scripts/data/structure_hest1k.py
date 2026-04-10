"""Download HEST1k samples matching a config filter."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.download import (
    create_structured_metadata_symlink,
    create_structured_symlinks,
    download_hest_metadata,
    download_sample,
    validate_hest_sample_mpp,
)
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.utils.getters import build_pipeline_config, resolve_samples


def get_hest_metadata_path(raw_dir: Path) -> Path:
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    if metadata_path.exists():
        return metadata_path
    return download_hest_metadata(raw_dir)


def ensure_hest_sample_downloaded(sample_id: str, raw_dir: Path) -> None:
    wsi_files = list((raw_dir / 'wsis').glob(f'{sample_id}*'))
    tx_files = list((raw_dir / 'transcripts').glob(f'{sample_id}*'))
    if len(wsi_files) == 1 and len(tx_files) == 1:
        return
    download_sample(sample_id, raw_dir)


def main(data_cfg: DataConfig) -> None:
    assert data_cfg.name == "hest1k", f"Expected dataset='hest1k', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)
    metadata_csv = get_hest_metadata_path(cfg.raw_dir)
    create_structured_metadata_symlink(metadata_csv, cfg.paths.structured_dir)
    samples = resolve_samples(cfg, metadata_csv)
    for sample_id in samples:
        ensure_hest_sample_downloaded(sample_id, cfg.raw_dir)
        validate_hest_sample_mpp(sample_id, cfg.raw_dir, metadata_csv)
        create_structured_symlinks(sample_id, cfg.raw_dir, cfg.paths.structured_dir)


def cli(argv: list[str] | None = None) -> int:
    data_cfg, _, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
