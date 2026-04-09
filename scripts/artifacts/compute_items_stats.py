"""Compute statistics for the items artifact defined by an artifacts config."""

from dotenv import load_dotenv

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.pipeline import compute_items_stats, plot_items_stats
from xenium_hne_fusion.processing_cli import parse_artifacts_args
from xenium_hne_fusion.utils.getters import get_managed_paths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
    batch_size: int = 256,
    num_workers: int = 10,
) -> None:
    load_dotenv()
    output_dir = get_managed_paths(artifacts_cfg.name).output_dir
    items_path = output_dir / 'items' / f'{artifacts_cfg.items.name}.json'
    assert items_path.exists(), f'Items not found: {items_path}'
    compute_items_stats(
        items_path,
        output_dir,
        overwrite=overwrite,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == '__main__':
    artifacts_cfg, overwrite_arg = parse_artifacts_args()
    main(artifacts_cfg, overwrite=overwrite_arg)
