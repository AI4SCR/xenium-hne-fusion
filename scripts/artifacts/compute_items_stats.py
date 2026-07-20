"""Compute statistics for the items artifact defined by an artifacts config."""

import sys
from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.stats import compute_items_stats
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(
    artifacts_cfg: ArtifactsConfig,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 10,
    output_path: Path | None = None,
) -> None:
    load_dotenv()
    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    assert items_path.exists(), f'Items not found: {items_path}'
    resolved_stats_path = managed_paths.resolve_statistics_path(output_path) if output_path is not None else None
    compute_items_stats(
        items_path,
        managed_paths,
        cell_type_col=artifacts_cfg.cell_type_col,
        overwrite=overwrite,
        batch_size=batch_size,
        num_workers=num_workers,
        stats_path=resolved_stats_path,
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--output-path', type=Path | None, default=None)

    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(
        init.artifacts,
        overwrite=init.overwrite,
        batch_size=init.batch_size,
        num_workers=init.num_workers,
        output_path=init.output_path,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
