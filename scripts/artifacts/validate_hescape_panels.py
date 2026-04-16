"""Validate HESCAPE panels against the feature universe of their splits."""

from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.hvg import load_feature_universe
from xenium_hne_fusion.utils.getters import get_managed_paths

PANELS_DIR = Path('panels/hescape')


def validate_hescape_panels() -> None:
    managed_paths = get_managed_paths('hest1k')

    for panel_yaml in sorted(PANELS_DIR.glob('*.yaml')):
        name = panel_yaml.stem
        split_path = managed_paths.output_dir / 'splits' / 'hescape' / name / 'hescape.parquet'
        assert split_path.exists(), f'Split not found: {split_path}'

        panel = yaml.load(panel_yaml.open(), yaml.SafeLoader)
        target_panel = panel['target_panel']
        source_panel = panel['source_panel']

        sample_ids = pd.read_parquet(split_path, columns=['sample_id'])['sample_id'].unique().tolist()
        gene_sets = [
            set(load_feature_universe(managed_paths.processed_dir / sid / 'feature_universe.txt'))
            for sid in sample_ids
        ]
        universe = set.intersection(*gene_sets)

        target_set = set(target_panel)
        source_set = set(source_panel)

        target_not_in_universe = target_set - universe
        source_not_in_universe = source_set - universe
        overlap = target_set & source_set

        assert not target_not_in_universe, f'[{name}] target genes outside universe: {sorted(target_not_in_universe)}'
        assert not source_not_in_universe, f'[{name}] source genes outside universe: {sorted(source_not_in_universe)}'
        assert not overlap, f'[{name}] target/source overlap: {sorted(overlap)}'

        logger.info(
            f'{name}: OK | universe={len(universe)} | '
            f'target={len(target_panel)} | source={len(source_panel)} | '
            f'uncovered={len(universe) - len(target_set) - len(source_set)}'
        )


def cli() -> int:
    auto_cli(validate_hescape_panels)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
