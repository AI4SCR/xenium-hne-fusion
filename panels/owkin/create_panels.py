"""Build per-sample-set source panels for owkin training.

For each artifacts config, intersects the per-sample gene universes
(`feature_universe.txt`) across that config's `items.filter.include_ids`, then writes a
panel YAML with the intersection as `source_panel` and an empty `target_panel` to
`DATA_DIR/03_output/owkin/panels/<items.name>.yaml`.

Usage:
    uv run python panels/owkin/create_panels.py
    uv run python panels/owkin/create_panels.py --config-dir configs/artifacts/owkin --pattern "*_cells.yaml" --overwrite true
"""
from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.artifacts.config import build_artifacts_parser
from xenium_hne_fusion.artifacts.panel import intersect_gene_universes, save_source_panel
from xenium_hne_fusion.utils.getters import ManagedPaths

load_dotenv(override=True)

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'artifacts' / 'owkin'


def main(config_paths: list[Path], overwrite: bool = False) -> None:
    assert config_paths, 'No config paths provided'

    for config_path in config_paths:
        parser = build_artifacts_parser()
        cfg = parser.instantiate(parser.parse_args(['--config', str(config_path)])).artifacts

        include_ids = cfg.items.filter.include_ids
        assert include_ids, f'{config_path}: items.filter.include_ids must be set'

        managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
        source_panel = intersect_gene_universes(include_ids, managed.processed_dir)
        output_path = managed.panels_dir / f'{cfg.items.name}.yaml'
        save_source_panel(output_path, source_panel, overwrite=overwrite)


def cli(argv: list[str] | None = None) -> int:
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config-dir', type=Path, default=DEFAULT_CONFIGS_DIR)
    parser.add_argument('--pattern', type=str, default='*_cells.yaml')
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args(argv)

    config_paths = sorted(args.config_dir.glob(args.pattern))
    assert config_paths, f'No configs matching {args.pattern!r} found in {args.config_dir}'
    main(config_paths, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    import sys

    raise SystemExit(cli(sys.argv[1:]))
