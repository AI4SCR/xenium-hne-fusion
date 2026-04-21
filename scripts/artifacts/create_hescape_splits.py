"""Materialize fixed HESCAPE sample splits from HEST1K all.json."""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import join_items_with_metadata
from xenium_hne_fusion.utils.getters import get_managed_paths

# Each panel maps to an explicit list of outer-fold dicts (fit/val/test sample IDs).
# Index in list == outer fold number (outer=0, outer=1, ...).
PANEL_TO_SPLITS: dict[str, list[dict[str, list[str]]]] = {
    'breast': [
        # outer=0: hardcoded holdout
        {'fit': ['TENX94', 'TENX95', 'TENX99'], 'val': ['NCBI785'], 'test': ['NCBI783']},
        # outer=1: swap val/test
        {'fit': ['TENX94', 'TENX95', 'TENX99'], 'val': ['NCBI783'], 'test': ['NCBI785']},
        # outer=2..4: rotate one fit sample into test
        {'fit': ['NCBI783', 'TENX95', 'TENX99'], 'val': ['NCBI785'], 'test': ['TENX94']},
        {'fit': ['NCBI783', 'TENX94', 'TENX99'], 'val': ['NCBI785'], 'test': ['TENX95']},
        {'fit': ['NCBI783', 'TENX94', 'TENX95'], 'val': ['NCBI785'], 'test': ['TENX99']},
    ],
    'bowel': [
        # outer=0: hardcoded holdout
        {'fit': ['TENX147', 'TENX148', 'TENX149'], 'val': ['TENX114'], 'test': ['TENX111']},
        # outer=1: swap val/test
        {'fit': ['TENX147', 'TENX148', 'TENX149'], 'val': ['TENX111'], 'test': ['TENX114']},
        # outer=2..4: rotate one fit sample into test
        {'fit': ['TENX111', 'TENX148', 'TENX149'], 'val': ['TENX114'], 'test': ['TENX147']},
        {'fit': ['TENX111', 'TENX147', 'TENX149'], 'val': ['TENX114'], 'test': ['TENX148']},
        {'fit': ['TENX111', 'TENX147', 'TENX148'], 'val': ['TENX114'], 'test': ['TENX149']},
    ],
    'human-immuno-oncology': [
        # outer=0: hardcoded holdout
        {'fit': ['TENX138', 'TENX139', 'TENX142'], 'val': ['TENX141'], 'test': ['TENX140']},
        # outer=1..2: rotate fit samples into test
        {'fit': ['TENX139', 'TENX140', 'TENX142'], 'val': ['TENX141'], 'test': ['TENX138']},
        {'fit': ['TENX138', 'TENX140', 'TENX142'], 'val': ['TENX141'], 'test': ['TENX139']},
        # outer=3: val conflicts with test -> use former test as val
        {'fit': ['TENX138', 'TENX139', 'TENX142'], 'val': ['TENX140'], 'test': ['TENX141']},
        # outer=4
        {'fit': ['TENX138', 'TENX139', 'TENX140'], 'val': ['TENX141'], 'test': ['TENX142']},
    ],
    # lung-healthy: 20 samples, replace all 4 holdout samples at once.
    # outer=0/1 match the external hescape CSV splits (test_0.csv / test_1.csv).
    # outer=2..4 rotate blocks of 4 fit samples into test.
    'lung-healthy': [
        # outer=0: test = external test_0.csv
        {
            'fit': [
                'NCBI856', 'NCBI859', 'NCBI860', 'NCBI864',
                'NCBI865', 'NCBI867', 'NCBI870', 'NCBI873',
                'NCBI875', 'NCBI876', 'NCBI879', 'NCBI880',
            ],
            'val': ['NCBI857', 'NCBI861', 'NCBI881', 'NCBI883'],
            'test': ['NCBI858', 'NCBI866', 'NCBI882', 'NCBI884'],
        },
        # outer=1: test = external test_1.csv
        {
            'fit': [
                'NCBI856', 'NCBI859', 'NCBI860', 'NCBI864',
                'NCBI865', 'NCBI867', 'NCBI870', 'NCBI873',
                'NCBI875', 'NCBI876', 'NCBI879', 'NCBI880',
            ],
            'val': ['NCBI858', 'NCBI866', 'NCBI882', 'NCBI884'],
            'test': ['NCBI857', 'NCBI861', 'NCBI881', 'NCBI883'],
        },
        # outer=2..4: rotate blocks of 4 fit samples into test
        {
            'fit': [
                'NCBI857', 'NCBI861', 'NCBI865', 'NCBI867',
                'NCBI870', 'NCBI873', 'NCBI875', 'NCBI876',
                'NCBI879', 'NCBI880', 'NCBI881', 'NCBI883',
            ],
            'val': ['NCBI858', 'NCBI866', 'NCBI882', 'NCBI884'],
            'test': ['NCBI856', 'NCBI859', 'NCBI860', 'NCBI864'],
        },
        {
            'fit': [
                'NCBI856', 'NCBI857', 'NCBI859', 'NCBI860',
                'NCBI861', 'NCBI864', 'NCBI875', 'NCBI876',
                'NCBI879', 'NCBI880', 'NCBI881', 'NCBI883',
            ],
            'val': ['NCBI858', 'NCBI866', 'NCBI882', 'NCBI884'],
            'test': ['NCBI865', 'NCBI867', 'NCBI870', 'NCBI873'],
        },
        {
            'fit': [
                'NCBI856', 'NCBI857', 'NCBI859', 'NCBI860',
                'NCBI861', 'NCBI864', 'NCBI865', 'NCBI867',
                'NCBI870', 'NCBI873', 'NCBI881', 'NCBI883',
            ],
            'val': ['NCBI858', 'NCBI866', 'NCBI882', 'NCBI884'],
            'test': ['NCBI875', 'NCBI876', 'NCBI879', 'NCBI880'],
        },
    ],
    'human-multi-tissue': [
        # outer=0: hardcoded holdout
        {
            'fit': ['TENX105', 'TENX118', 'TENX119', 'TENX121', 'TENX123', 'TENX126', 'TENX132', 'TENX133', 'TENX134'],
            'val': ['TENX106', 'TENX120', 'TENX125'],
            'test': ['TENX116', 'TENX122', 'TENX124'],
        },
        # outer=1: TENX106 conflicts with val -> replace with TENX124
        {
            'fit': ['TENX116', 'TENX119', 'TENX121', 'TENX122', 'TENX123', 'TENX126', 'TENX132', 'TENX133', 'TENX134'],
            'val': ['TENX120', 'TENX124', 'TENX125'],
            'test': ['TENX105', 'TENX106', 'TENX118'],
        },
        # outer=2: TENX120 conflicts with val -> replace with TENX124
        {
            'fit': ['TENX105', 'TENX116', 'TENX118', 'TENX122', 'TENX123', 'TENX126', 'TENX132', 'TENX133', 'TENX134'],
            'val': ['TENX106', 'TENX124', 'TENX125'],
            'test': ['TENX119', 'TENX120', 'TENX121'],
        },
        # outer=3: TENX125 conflicts with val -> replace with TENX124
        {
            'fit': ['TENX105', 'TENX116', 'TENX118', 'TENX119', 'TENX121', 'TENX122', 'TENX132', 'TENX133', 'TENX134'],
            'val': ['TENX106', 'TENX120', 'TENX124'],
            'test': ['TENX123', 'TENX125', 'TENX126'],
        },
        # outer=4: no val conflicts
        {
            'fit': ['TENX105', 'TENX116', 'TENX118', 'TENX119', 'TENX121', 'TENX122', 'TENX123', 'TENX124', 'TENX126'],
            'val': ['TENX106', 'TENX120', 'TENX125'],
            'test': ['TENX132', 'TENX133', 'TENX134'],
        },
    ],
}


def _assign_splits(
    filtered: pd.DataFrame,
    fit_samples: list[str],
    val_samples: list[str],
    test_samples: list[str],
) -> pd.DataFrame:
    filtered = filtered.copy()
    filtered['split'] = None
    for label, samples in [('fit', fit_samples), ('val', val_samples), ('test', test_samples)]:
        filtered.loc[filtered['sample_id'].isin(samples), 'split'] = label

    assert filtered['split'].notna().all(), 'Some tiles are missing split labels'
    assert filtered.index.is_unique, 'Tile-level metadata index must be unique'

    dtype = pd.CategoricalDtype(categories=['fit', 'val', 'test'], ordered=False)
    filtered['split'] = filtered['split'].astype(dtype)
    return filtered


def _create_outer_split_files(
    name: str,
    outer_splits: list[dict[str, list[str]]],
    joined: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> list[Path]:
    all_sample_ids = {s for fold in outer_splits for split in fold.values() for s in split}
    filtered = joined.loc[joined['sample_id'].isin(all_sample_ids)].copy()

    present = set(filtered['sample_id'])
    missing = sorted(all_sample_ids - present)
    assert not missing, f'Split samples missing from items: {missing}'

    output_paths = []
    for outer, fold in enumerate(outer_splits):
        output_path = output_dir / 'splits' / 'hescape' / name / f'outer={outer}-seed=0.parquet'
        if output_path.exists():
            assert overwrite, f'Split already exists: {output_path}'

        all_in_fold = set(fold['fit']) | set(fold['val']) | set(fold['test'])
        assert len(all_in_fold) == len(fold['fit']) + len(fold['val']) + len(fold['test']), (
            f'Duplicate samples across splits in outer={outer} for {name}'
        )
        assert all_in_fold == all_sample_ids, (
            f'Sample mismatch in outer={outer} for {name}: '
            f'expected {sorted(all_sample_ids)}, got {sorted(all_in_fold)}'
        )

        df = _assign_splits(filtered, fit_samples=fold['fit'], val_samples=fold['val'], test_samples=fold['test'])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        output_paths.append(output_path)

    return output_paths


def create_hescape_splits(overwrite: bool = False) -> list[Path]:
    managed_paths = get_managed_paths('hest1k')
    items_path = managed_paths.output_dir / 'items' / 'all.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet'

    assert items_path.exists(), f'Items not found: {items_path}'
    assert metadata_path.exists(), f'Metadata not found: {metadata_path}'

    joined = join_items_with_metadata(items_path, metadata_path)
    output_paths = []
    for name, outer_splits in PANEL_TO_SPLITS.items():
        output_paths += _create_outer_split_files(
            name=name,
            outer_splits=outer_splits,
            joined=joined,
            output_dir=managed_paths.output_dir,
            overwrite=overwrite,
        )
    return output_paths


def cli() -> int:
    auto_cli(create_hescape_splits)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
