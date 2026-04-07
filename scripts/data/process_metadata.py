"""Clean dataset metadata into a canonical sample-level parquet."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import get_structured_metadata_path, process_dataset_metadata
from xenium_hne_fusion.utils.getters import load_pipeline_config, resolve_samples


def main(dataset: str, config_path: Path | None = None) -> None:
    cfg = load_pipeline_config(dataset, config_path)
    metadata_path = get_structured_metadata_path(cfg.paths.structured_dir)
    sample_ids = cfg.processing.filter.sample_ids if cfg.processing.name == 'beat' else resolve_samples(cfg, metadata_path)
    process_dataset_metadata(
        dataset=cfg.processing.name,
        metadata_path=metadata_path,
        output_path=cfg.paths.processed_dir / 'metadata.parquet',
        sample_ids=sample_ids,
    )


if __name__ == '__main__':
    auto_cli(main)
