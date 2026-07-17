"""Structure raw owkin data into the 01_structured canonical layout."""

from loguru import logger

from xenium_hne_fusion.data.config import DataConfig
from xenium_hne_fusion.structure import structure_metadata, structure_sample
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(config: DataConfig, *, overwrite: bool = False) -> int:
    paths = ManagedPaths(data_dir=config.data_dir, name=config.name, raw_dir=config.raw_dir)
    structured_dir = paths.structured_dir
    raw_dir = config.raw_dir

    metadata_path = raw_dir / "matched_samples" / "matched_pairs.parquet"
    if metadata_path.exists():
        structure_metadata(metadata_path, structured_dir)

    sample_dirs = sorted(
        path
        for path in (raw_dir / "matched_samples_processed").iterdir()
        if path.is_dir() and path.name.startswith("CH_")
    )
    logger.info(f"Found {len(sample_dirs)} owkin samples in {raw_dir}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        if (structured_dir / sample_id).exists() and not overwrite:
            logger.info(f"Skipping already-structured {sample_id}")
            continue

        wsi_path = sample_dir / "region.tif"
        assert wsi_path.exists(), f"{wsi_path} does not exist"

        tx_path = sample_dir / "transcripts" / "transcripts.parquet"
        assert tx_path.exists(), f"{tx_path} does not exist"

        cell_features_dir = (
            raw_dir / "matched_samples" / sample_id / "xenium/normalised_results/outs" / "cell_feature_matrix"
        )
        assert cell_features_dir.exists(), f"{cell_features_dir} does not exist"

        cells_path = raw_dir / "matched_samples_cells_aligned" / sample_id / "centroids/centroid_cropped.parquet"
        assert cells_path.exists(), f"{cells_path} does not exist"

        structure_sample(
            sample_id,
            wsi_path,
            tx_path,
            structured_dir,
            cells_path=cells_path,
            cell_features_dir=cell_features_dir,
        )
    return 0


def cli(argv: list[str] | None = None) -> int:
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True)
    parser.add_class_arguments(DataConfig, nested_key="data")
    parser.add_argument("--overwrite", type=bool, default=False)

    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.data, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
