import json
from pathlib import Path

import pandas as pd
import yaml

from xenium_hne_fusion.artifacts.panel import build_panel
from xenium_hne_fusion.datasets.tiles import TileDataset


def _write_feature_universe(sample_dir: Path, genes: list[str]) -> None:
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


def test_build_panel_saves_intersection_across_sample_ids(tmp_path: Path):
    processed_dir = tmp_path / 'processed'
    _write_feature_universe(processed_dir / 'S1', ['A', 'B', 'C'])
    _write_feature_universe(processed_dir / 'S2', ['B', 'C', 'D'])

    items_path = tmp_path / 'items.json'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/a'},
                {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': '/tmp/b'},
            ]
        )
    )
    output_path = tmp_path / 'panels' / 'default.yaml'

    build_panel(items_path, processed_dir, output_path)

    panel = yaml.safe_load(output_path.read_text())
    assert panel['source_panel'] == ['B', 'C']
    assert panel['target_panel'] == []
