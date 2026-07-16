import json
from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.artifacts.panel import create_panel, get_common_genes
from xenium_hne_fusion.datasets.tiles import TileDataset


def _write_expr_parquet(tile_dir: Path, genes: list[str], rows: list[list[int]]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=genes).to_parquet(tile_dir / 'expr-kernel_size=16.parquet', index=False)


def _write_feature_universe(tile_dir: Path, genes: list[str]) -> None:
    sample_dir = tile_dir.parent.parent
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / 'feature_universe.txt').write_text('\n'.join(genes) + '\n')


def test_tile_dataset_filters_to_fit_split(tmp_path: Path):
    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/a'},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/b'},
                {'id': 'S1_2', 'sample_id': 'S1', 'tile_id': 2, 'tile_dir': '/tmp/c'},
            ]
        )
    )
    pd.DataFrame(
        {'split': ['fit', 'val', 'test']},
        index=pd.Index(['S1_0', 'S1_1', 'S1_2'], name='id'),
    ).to_parquet(split_path)

    ds = TileDataset(
        target='expression',
        source_panel=[],
        target_panel=[],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        metadata_path=split_path,
        split='fit',
        id_key='id',
    )
    ds.setup()

    assert [item['id'] for item in ds.items] == ['S1_0']


def test_get_common_genes_uses_intersection_across_samples(tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S2' / '256_256' / '0'
    _write_expr_parquet(tile_a, ['A', 'B', 'C'], [[1, 0, 0]])
    _write_expr_parquet(tile_b, ['B', 'C', 'D'], [[1, 0, 0]])
    _write_feature_universe(tile_a, ['A', 'B', 'C'])
    _write_feature_universe(tile_b, ['B', 'C', 'D'])

    fit_items = pd.DataFrame(
        [
            {'id': 'S1_0', 'sample_id': 'S1', 'split': 'fit'},
            {'id': 'S2_0', 'sample_id': 'S2', 'split': 'fit'},
        ]
    )

    assert get_common_genes(fit_items, processed_dir=tmp_path) == ['B', 'C']


def test_create_panel_rejects_when_common_genes_are_fewer_than_requested(tmp_path: Path):
    tile_dir = tmp_path / 'S1' / '256_256' / '0'
    _write_expr_parquet(tile_dir, ['A'], [[1], [0]])
    _write_feature_universe(tile_dir, ['A'])

    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    output_path = tmp_path / 'panels' / 'hvg-expr.yaml'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_dir)},
            ]
        )
    )
    pd.DataFrame(
        {
            'split': ['fit'],
            'sample_id': ['S1'],
        },
        index=pd.Index(['S1_0'], name='id'),
    ).to_parquet(split_path)

    with pytest.raises(AssertionError, match='exceeds common genes'):
        create_panel(
            items_path=items_path,
            split_metadata_path=split_path,
            processed_dir=tmp_path,
            output_path=output_path,
            n_top_genes=2,
            overwrite=True,
        )


