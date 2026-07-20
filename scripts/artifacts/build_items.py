"""Collect processed tiles into the source items list, optionally computing item stats."""

import sys

from dotenv import load_dotenv

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.items import create_items
from xenium_hne_fusion.artifacts.stats import compute_items_stats
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(artifacts_cfg: ArtifactsConfig, *, overwrite: bool = False, compute_stats: bool = True) -> None:
    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    items_path = create_items(
        managed_paths.items_dir,
        managed_paths.processed_dir,
        tile_px=artifacts_cfg.tile_px,
        stride_px=artifacts_cfg.stride_px,
        overwrite=overwrite,
    )
    if compute_stats:
        compute_items_stats(
            items_path,
            managed_paths,
            cell_type_col=artifacts_cfg.cell_type_col,
            overwrite=overwrite,
        )


def cli(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_artifacts_parser()
    parser.add_argument('--compute-stats', type=bool, default=True)
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.artifacts, overwrite=init.overwrite, compute_stats=init.compute_stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
