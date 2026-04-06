from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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
