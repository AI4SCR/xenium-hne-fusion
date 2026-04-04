"""Clean dataset metadata into a canonical sample-level parquet."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import clean_sample_metadata, get_structured_metadata_path
from xenium_hne_fusion.utils.getters import load_pipeline_config, resolve_samples


def main(dataset: str, config_path: Path | None = None) -> None:
    cfg = load_pipeline_config(dataset, config_path)
    raw_metadata_path = get_structured_metadata_path(cfg.structured_dir)
    sample_ids = resolve_samples(cfg, raw_metadata_path)
    clean_sample_metadata(
        metadata_path=raw_metadata_path,
        output_path=cfg.processed_dir / 'metadata.parquet',
        sample_ids=sample_ids,
    )


if __name__ == '__main__':
    auto_cli(main)
