"""Clean dataset metadata into a canonical sample-level parquet."""

import sys

from dotenv import load_dotenv

load_dotenv()

from xenium_hne_fusion.config import ProcessingConfig
from xenium_hne_fusion.metadata import (
    get_structured_metadata_path,
    normalize_sample_metadata,
    process_dataset_metadata,
    read_metadata_table,
)
from xenium_hne_fusion.processing_cli import parse_processing_args
from xenium_hne_fusion.utils.getters import build_pipeline_config, resolve_samples, select_sample_ids


def main(processing_cfg: ProcessingConfig) -> None:
    cfg = build_pipeline_config(processing_cfg)
    dataset = cfg.dataset
    metadata_path = get_structured_metadata_path(cfg.paths.structured_dir)
    if dataset == 'hest1k':
        selected_sample_ids = resolve_samples(cfg, metadata_path)
    else:
        metadata = read_metadata_table(metadata_path)
        if 'sample_id' not in metadata.columns:
            assert metadata.index.name == 'sample_id', 'BEAT metadata must use sample_id index'
            metadata = metadata.reset_index()
        metadata = normalize_sample_metadata(metadata)
        selected_sample_ids = select_sample_ids(metadata['sample_id'].tolist(), cfg.processing.filter)
    process_dataset_metadata(
        dataset=cfg.processing.name,
        metadata_path=metadata_path,
        output_path=cfg.paths.processed_dir / 'metadata.parquet',
        selected_sample_ids=selected_sample_ids,
    )


if __name__ == '__main__':
    processing_cfg, _, _, _ = parse_processing_args(sys.argv[1:], include_executor=False)
    main(processing_cfg)
