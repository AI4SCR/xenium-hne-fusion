"""Download HEST1k samples matching a config filter."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.download import create_raw_symlinks, download_sample
from xenium_hne_fusion.utils.getters import load_config, resolve_samples


def main(pipeline_config: Path) -> None:
    cfg = load_config(pipeline_config)
    samples = resolve_samples(cfg)
    for sample_id in samples:
        download_sample(sample_id, cfg.download_dir)
        create_raw_symlinks(sample_id, cfg.download_dir, cfg.raw_dir)


if __name__ == "__main__":
    auto_cli(main)
