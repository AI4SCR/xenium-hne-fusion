"""Derive HESCAPE output panels from split feature universes and fixed target panels."""

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


def create_hescape_panels(overwrite: bool = False) -> list[Path]:
    managed_paths = get_managed_paths('hest1k')
    output_paths = []

    for panel_yaml in sorted(PANELS_DIR.glob('*.yaml')):
        name = panel_yaml.stem
        split_path = managed_paths.output_dir / 'splits' / 'hescape' / name / 'outer=0-seed=0.parquet'
        assert split_path.exists(), f'Split not found: {split_path}'

        target_panel = yaml.safe_load(panel_yaml.read_text())['target_panel']

        split_metadata = pd.read_parquet(split_path, columns=['sample_id'])
        sample_ids = split_metadata['sample_id'].unique().tolist()

        gene_sets = [
            set(load_feature_universe(managed_paths.processed_dir / sid / 'feature_universe.txt'))
            for sid in sample_ids
        ]
        common = sorted(set.intersection(*gene_sets))
        common_set = set(common)
        target_set = set(target_panel)

        missing = target_set - common_set
        assert not missing, f'[{name}] target genes missing from feature universe: {missing}'

        source_panel = [g for g in common if g not in target_set]

        logger.info(
            f'{name}: {len(common)} common genes | '
            f'{len(target_panel)} target | {len(source_panel)} source | '
            f'{len(sample_ids)} samples'
        )

        output_path = managed_paths.output_dir / 'panels' / 'hescape' / f'{name}.yaml'
        if output_path.exists():
            assert overwrite, f'Panel already exists: {output_path}'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump({'source_panel': source_panel, 'target_panel': target_panel}, sort_keys=False)
        )
        output_paths.append(output_path)

    return output_paths


def cli() -> int:
    auto_cli(create_hescape_panels)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
