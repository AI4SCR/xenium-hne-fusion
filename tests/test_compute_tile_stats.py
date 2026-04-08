import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.data import compute_tile_stats as module
from scripts.data.compute_tile_stats import _compute_item_stats


def test_compute_item_stats_reads_tile_local_cells_parquet(tmp_path: Path):
    tile_dir = tmp_path / '0'
    tile_dir.mkdir(parents=True, exist_ok=True)

    transcripts = gpd.GeoDataFrame(
        {'feature_name': ['A', 'A', 'B']},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    cells = gpd.GeoDataFrame(
        {'Level3_grouped': ['tumor', 'stroma', 'tumor']},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    transcripts.to_parquet(tile_dir / 'transcripts.parquet')
    cells.to_parquet(tile_dir / 'cells.parquet')

    stats = _compute_item_stats(
        {'id': 'S1_0', 'tile_dir': str(tile_dir)},
        cell_type_col='Level3_grouped',
    )

    assert stats == {
        'id': 'S1_0',
        'num_transcripts': 3,
        'num_unique_transcripts': 2,
        'num_cells': 3,
        'num_unique_cells': 2,
    }


def test_plot_tile_stats_writes_transcript_scatter_plots(tmp_path: Path):
    stats = pd.DataFrame(
        {
            'num_transcripts': [10, 100, 1000],
            'num_unique_transcripts': [5, 20, 100],
            'num_cells': [1, 2, 3],
            'num_unique_cells': [1, 2, 2],
        },
        index=['a', 'b', 'c'],
    )

    module.plot_tile_stats(stats, tmp_path)

    assert (tmp_path / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (tmp_path / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_main_accepts_explicit_items_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / '03_output' / 'hest1k'
    tile_dir = tmp_path / 'tiles' / '0'
    tile_dir.mkdir(parents=True)
    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True)
    items_path.write_text(json.dumps([
        {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': '0',
            'tile_dir': str(tile_dir),
        }
    ]))

    cfg = SimpleNamespace(
        paths=SimpleNamespace(output_dir=output_dir),
        processing=SimpleNamespace(name='hest1k'),
    )
    monkeypatch.setattr(module, 'load_pipeline_config', lambda dataset, config_path: cfg)
    monkeypatch.setattr(
        module,
        '_compute_item_stats',
        lambda item, cell_type_col: {
            'id': item['id'],
            'num_transcripts': 10,
            'num_unique_transcripts': 5,
            'num_cells': 3,
            'num_unique_cells': 2,
        },
    )

    module.main(dataset='hest1k', items_name=None, items_path=items_path)

    assert (output_dir / 'statistics' / 'subset.parquet').exists()
    figures_dir = output_dir / 'figures' / 'tile_stats' / 'subset'
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()
