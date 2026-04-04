"""Download HEST1k samples matching a config filter."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.download import (
    create_structured_metadata_symlink,
    create_structured_symlinks,
    download_hest_metadata,
    download_sample,
)
from xenium_hne_fusion.utils.getters import load_pipeline_config, resolve_samples


def main(dataset: str, config_path: Path | None = None) -> None:
    cfg = load_pipeline_config(dataset, config_path)
    metadata_csv = download_hest_metadata(cfg.raw_dir)
    create_structured_metadata_symlink(metadata_csv, cfg.structured_dir)
    samples = resolve_samples(cfg, metadata_csv)
    for sample_id in samples:
        download_sample(sample_id, cfg.raw_dir)
        create_structured_symlinks(sample_id, cfg.raw_dir, cfg.structured_dir)


if __name__ == "__main__":
    auto_cli(main)
