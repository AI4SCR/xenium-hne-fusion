"""Structure raw BEAT data into 01_structured canonical layout."""

import sys

from dotenv import load_dotenv
from loguru import logger

from xenium_hne_fusion.config import DataConfig, FilterConfig, TilesConfig
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.structure import structure_metadata, structure_sample
from xenium_hne_fusion.utils.getters import build_pipeline_config
import scanpy as sc
from pathlib import Path

data_cfg = DataConfig(name="owkin", filter=FilterConfig(), tiles=TilesConfig(tile_px=512, stride_px=512, mpp=.5, img_size=224))

def main(data_cfg: DataConfig) -> None:
    load_dotenv()
    assert data_cfg.name == "owkin", f"Expected dataset='owkin', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)

    # TODO: pull in real metadata
    metadata_path = cfg.raw_dir / "matched_samples" / "matched_pairs.parquet"
    if metadata_path.exists():
        structure_metadata(metadata_path, cfg.paths.structured_dir)

    sample_dirs = sorted(path for path in (cfg.raw_dir / 'matched_samples_processed').iterdir() if path.is_dir() and path.name.startswith('CH_'))
    logger.info(f"Found {len(sample_dirs)} samples in {cfg.raw_dir}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        wsi_path = sample_dir / "region.tif"
        assert wsi_path.exists(), f'{wsi_path} does not exist'

        tx_path = sample_dir / 'transcripts' / "transcripts.parquet"
        assert tx_path.exists(), f'{tx_path} does not exist'

        cell_features_dir = cfg.raw_dir / 'matched_samples' / sample_id / 'xenium/normalised_results/outs' / "cell_feature_matrix"
        assert cell_features_dir.exists(), f'{cell_features_dir} does not exist'

        cells_path = (
                cfg.raw_dir
                / "matched_samples_cells_aligned"
                / sample_id
                / "centroids/centroid_cropped.parquet"
        )
        assert cells_path.exists(), f'{cells_path} does not exist'

        structure_sample(
            sample_id,
            wsi_path,
            tx_path,
            cfg.paths.structured_dir,
            cells_path=cells_path if cells_path.exists() else None,
            cell_features_dir=cell_features_dir,
            visualize=False,
        )

# %%
def cli(argv: list[str] | None = None) -> int:
    data_cfg, _, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
