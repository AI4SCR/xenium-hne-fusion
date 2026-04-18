"""Materialize fixed HESCAPE sample splits from HEST1K all.json."""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import join_items_with_metadata
from xenium_hne_fusion.utils.getters import get_managed_paths

PANEL_TO_SPLITS = {
    # 'human-5k': {
    #     'fit': {
    #         4: 'Xenium_Prime_Human_Lung_Cancer_FFPE',
    #         2: 'Xenium_Prime_Breast_Cancer_FFPE',
    #         3: 'Xenium_Prime_Cervical_Cancer_FFPE',
    #         5: 'Xenium_Prime_Human_Ovary_FF',
    #     },
    #     'val': {
    #         6: 'Xenium_Prime_Ovarian_Cancer_FFPE_XRrun',
    #     },
    #     'test': {
    #         1: 'TENX158',
    #         0: 'TENX157',
    #     },
    # },
    'breast': {
        'fit': {
            8: 'TENX99',
            4: 'TENX95',
            3: 'TENX94',
        },
        'val': {
            2: 'NCBI785',
        },
        'test': {
            0: 'NCBI783',
        },
    },
    'bowel': {
        'fit': {
            4: 'TENX149',
            3: 'TENX148',
            2: 'TENX147',
        },
        'val': {
            1: 'TENX114',
        },
        'test': {
            0: 'TENX111',
        },
    },
    'human-immuno-oncology': {
        'fit': {
            4: 'TENX142',
            1: 'TENX139',
            0: 'TENX138',
        },
        'val': {
            3: 'TENX141',
        },
        'test': {
            2: 'TENX140',
        },
    },
    'lung-healthy': {
        'fit': {
            15: 'NCBI880',
            14: 'NCBI879',
            13: 'NCBI876',
            12: 'NCBI875',
            11: 'NCBI873',
            10: 'NCBI870',
            9: 'NCBI867',
            7: 'NCBI865',
            6: 'NCBI864',
            4: 'NCBI860',
            3: 'NCBI859',
            0: 'NCBI856',
        },
        'val': {
            19: 'NCBI884',
            17: 'NCBI882',
            8: 'NCBI866',
            2: 'NCBI858',
        },
        'test': {
            18: 'NCBI883',
            16: 'NCBI881',
            5: 'NCBI861',
            1: 'NCBI857',
        },
    },
    'human-multi-tissue': {
        'fit': {
            14: 'TENX134',
            13: 'TENX133',
            12: 'TENX132',
            11: 'TENX126',
            8: 'TENX123',
            6: 'TENX121',
            4: 'TENX119',
            3: 'TENX118',
            0: 'TENX105',
        },
        'val': {
            10: 'TENX125',
            5: 'TENX120',
            1: 'TENX106',
        },
        'test': {
            9: 'TENX124',
            7: 'TENX122',
            2: 'TENX116',
        },
    },
}


def _get_outer_test_chunks(split_to_ids: dict[str, dict[int, str]]) -> list[list[str]]:
    """Return list of test-sample lists per outer fold.

    Chunk 0 = hardcoded test samples. Remaining samples (sorted by index) are
    split into equal-size chunks for folds 1..N.
    """
    all_indexed: dict[int, str] = {}
    for ids in split_to_ids.values():
        all_indexed.update(ids)

    test_samples = list(split_to_ids['test'].values())
    test_set = set(test_samples)
    chunk_size = len(test_samples)

    remaining = [all_indexed[k] for k in sorted(all_indexed) if all_indexed[k] not in test_set]
    assert len(remaining) % chunk_size == 0, (
        f'Remaining {len(remaining)} samples not divisible by chunk size {chunk_size}'
    )
    chunks = [remaining[i:i + chunk_size] for i in range(0, len(remaining), chunk_size)]
    return [test_samples] + chunks


def _resolve_val(
    val_samples: list[str],
    test_chunk: list[str],
    fallback_pool: list[str],
) -> list[str]:
    """Return val sample list for a fold, replacing any test-conflicts with fallback_pool samples."""
    test_set = set(test_chunk)
    conflicts = [s for s in val_samples if s in test_set]
    if not conflicts:
        return val_samples

    assert len(conflicts) <= len(fallback_pool), (
        f'Not enough fallback samples ({len(fallback_pool)}) to replace {len(conflicts)} conflicts'
    )
    replacements = fallback_pool[:len(conflicts)]
    resolved = [s for s in val_samples if s not in test_set] + replacements
    assert set(resolved).isdisjoint(test_set), 'Val still overlaps with test after resolution'
    return resolved


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
    split_to_ids: dict[str, dict[int, str]],
    joined: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> list[Path]:
    split_ids = {s for ids in split_to_ids.values() for s in ids.values()}
    filtered = joined.loc[joined['sample_id'].isin(split_ids)].copy()

    present = set(filtered['sample_id'])
    missing = sorted(split_ids - present)
    assert not missing, f'Split samples missing from items: {missing}'

    hardcoded_test = list(split_to_ids['test'].values())
    hardcoded_val = list(split_to_ids['val'].values())
    hardcoded_fit = list(split_to_ids['fit'].values())

    test_chunks = _get_outer_test_chunks(split_to_ids)
    all_samples = set(split_ids)

    output_paths = []
    for outer, test_chunk in enumerate(test_chunks):
        output_path = output_dir / 'splits' / 'hescape' / name / f'outer={outer}-seed=0.parquet'
        if output_path.exists():
            assert overwrite, f'Split already exists: {output_path}'

        if outer == 0:
            val = hardcoded_val
            fit = hardcoded_fit
        else:
            val = _resolve_val(hardcoded_val, test_chunk, fallback_pool=hardcoded_test)
            fit = sorted(all_samples - set(test_chunk) - set(val))

        df = _assign_splits(filtered, fit_samples=fit, val_samples=val, test_samples=test_chunk)

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
    for name, split_to_ids in PANEL_TO_SPLITS.items():
        output_paths += _create_outer_split_files(
            name=name,
            split_to_ids=split_to_ids,
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
