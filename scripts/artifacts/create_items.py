"""Collect processed tiles into the source items list for TileDataset."""

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.items import create_items
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(config: ArtifactsConfig, *, overwrite: bool = False) -> int:
    paths = ManagedPaths(data_dir=config.data_dir, name=config.name)
    create_items(
        paths.items_dir,
        paths.processed_dir,
        tile_px=config.tile_px,
        stride_px=config.stride_px,
        overwrite=overwrite,
    )
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.artifacts, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
