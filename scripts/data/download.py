"""Download HEST1k samples matching a config filter."""
from __future__ import annotations

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.download import create_structured_symlinks, download_sample
from xenium_hne_fusion.utils.getters import PipelineConfig, resolve_samples


def main(cfg: PipelineConfig) -> None:
    samples = resolve_samples(cfg)
    for sample_id in samples:
        download_sample(sample_id, cfg.raw_dir)
        create_structured_symlinks(sample_id, cfg.raw_dir, cfg.structured_dir)


if __name__ == "__main__":
    auto_cli(main)
