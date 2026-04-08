"""Clean dataset metadata into a canonical sample-level parquet."""

import sys

from dotenv import load_dotenv

load_dotenv()

from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.metadata import get_structured_metadata_path, process_dataset_metadata
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import build_pipeline_config, infer_dataset, resolve_samples


def main(processing_cfg: ProcessingConfig) -> None:
    cfg = build_pipeline_config(processing_cfg)
    dataset = infer_dataset(processing_cfg.name)
    metadata_path = get_structured_metadata_path(cfg.paths.structured_dir)
    sample_ids = cfg.processing.filter.sample_ids if dataset == 'beat' else resolve_samples(cfg, metadata_path)
    process_dataset_metadata(
        dataset=cfg.processing.name,
        metadata_path=metadata_path,
        output_path=cfg.paths.processed_dir / 'metadata.parquet',
        sample_ids=sample_ids,
    )


if __name__ == '__main__':
    processing_cfg, _, _ = parse_processing_args(sys.argv[1:], include_executor=False)
    main(processing_cfg)
