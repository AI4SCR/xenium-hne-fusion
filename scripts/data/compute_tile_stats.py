"""Compute per-tile statistics from an explicit items artifact."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

from xenium_hne_fusion.pipeline import compute_tile_stats_from_items, plot_tile_stats
from xenium_hne_fusion.utils.getters import load_pipeline_config

def main(
    dataset: str,
    items_path: Path,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 10,
) -> None:
    load_dotenv()
    if output_dir is None:
        output_dir = load_pipeline_config(dataset, config_path).paths.output_dir
    compute_tile_stats_from_items(
        items_path,
        output_dir,
        overwrite=overwrite,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == '__main__':
    auto_cli(main)
