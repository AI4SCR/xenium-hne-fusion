"""Materialize fixed HESCAPE sample splits from HEST1K all.json."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()

from xenium_hne_fusion.metadata import join_items_with_metadata
from xenium_hne_fusion.utils.getters import get_managed_paths

PANEL_TO_SPLITS = {
    # 'hescape-human-5k': {
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
    'hescape-breast': {
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
    'hescape-bowel': {
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
    'hescape-human-immuno-oncology': {
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
    'hescape-lung-healthy': {
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
    'hescape-human-multi-tissue': {
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


def _create_single_hescape_split(
    name: str,
    split_to_ids: dict[str, dict[int, str]],
    joined,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output_path = output_dir / 'splits' / 'hescape' / f'{name}.parquet'
    if output_path.exists():
        assert overwrite, f'Split already exists: {output_path}'

    split_ids = {sample_id for ids in split_to_ids.values() for sample_id in ids.values()}
    keep = joined['sample_id'].isin(split_ids)
    filtered = joined.loc[keep].copy()

    present = set(filtered['sample_id'])
    missing = sorted(split_ids - present)
    assert not missing, f'Split samples missing from items: {missing}'

    filtered['split'] = None
    for split_name, indexed_ids in split_to_ids.items():
        mask = filtered['sample_id'].isin(indexed_ids.values())
        filtered.loc[mask, 'split'] = split_name

    assert filtered.index.is_unique, 'Tile-level metadata index must be unique'
    assert filtered['split'].notna().all(), 'Some tiles are missing split labels'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path)
    return output_path


def create_hescape_splits(overwrite: bool = False) -> list[Path]:
    managed_paths = get_managed_paths('hest1k')
    items_path = managed_paths.output_dir / 'items' / 'all.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet'

    assert items_path.exists(), f'Items not found: {items_path}'
    assert metadata_path.exists(), f'Metadata not found: {metadata_path}'

    joined = join_items_with_metadata(items_path, metadata_path)
    output_paths = []
    for name, split_to_ids in PANEL_TO_SPLITS.items():
        output_paths.append(
            _create_single_hescape_split(
                name=name,
                split_to_ids=split_to_ids,
                joined=joined,
                output_dir=managed_paths.output_dir,
                overwrite=overwrite,
            )
        )
    return output_paths


def cli() -> int:
    auto_cli(create_hescape_splits)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
