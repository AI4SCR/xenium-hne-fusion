"""Write split parquets for a filtered item set from a hand-authored splits YAML."""

import sys

from dotenv import load_dotenv

from xenium_hne_fusion.artifacts.config import ArtifactsConfig, build_artifacts_parser
from xenium_hne_fusion.artifacts.splits import write_split_collection
from xenium_hne_fusion.utils.getters import ManagedPaths


def main(artifacts_cfg: ArtifactsConfig, overwrite: bool = False) -> None:
    managed_paths = ManagedPaths(data_dir=artifacts_cfg.data_dir, name=artifacts_cfg.name)
    items_path = managed_paths.items_dir / f'{artifacts_cfg.items.name}.json'
    assert items_path.exists(), f'Filtered items not found: {items_path}'
    write_split_collection(
        items_path=items_path,
        splits_yaml_path=artifacts_cfg.split.path,
        name=artifacts_cfg.split.name,
        output_dir=managed_paths.output_dir,
        overwrite=overwrite,
    )


def cli(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_artifacts_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.artifacts, overwrite=init.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli(sys.argv[1:]))
