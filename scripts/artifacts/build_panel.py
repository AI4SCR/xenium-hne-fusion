"""Build (or validate) the source panel for a filtered item set."""

import sys

from dotenv import load_dotenv
from loguru import logger

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.panel import build_panel
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(artifacts_cfg: ArtifactsConfig, *, overwrite: bool = False) -> None:
    panel_cfg = artifacts_cfg.panel
    assert panel_cfg is not None, 'panel is required'

    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    panel_path = managed_paths.panels_dir / f'{panel_cfg.name}.yaml'

    if panel_path.exists() and not overwrite:
        logger.info(f'Panel already exists: {panel_path}')
        return

    items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    assert items_path.exists(), f'Items not found: {items_path}'
    build_panel(items_path, managed_paths.processed_dir, panel_path, overwrite=overwrite)


def cli(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.artifacts, overwrite=init.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
