"""Structure raw BEAT data into 01_structured canonical layout."""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from xenium_hne_fusion.structure import structure_metadata, structure_sample
from xenium_hne_fusion.utils.getters import load_pipeline_config


def main(dataset: str = "beat", config_path: Path | None = None) -> None:
    load_dotenv()
    cfg = load_pipeline_config(dataset, config_path)

    metadata_path = cfg.raw_dir / "metadata.parquet"
    if metadata_path.exists():
        structure_metadata(metadata_path, cfg.structured_dir)

    sample_dirs = sorted(p for p in cfg.raw_dir.iterdir() if p.is_dir())
    logger.info(f"Found {len(sample_dirs)} samples in {cfg.raw_dir}")
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        wsi_path = next(
            (sample_dir / f for f in ("region.tiff", "region.tif") if (sample_dir / f).exists()),
            sample_dir / "region.tiff",
        )
        tx_path = sample_dir / "transcripts" / "transcripts.parquet"
        structure_sample(sample_id, wsi_path, tx_path, cfg.structured_dir)


if __name__ == "__main__":
    from jsonargparse import auto_cli
    auto_cli(main)
