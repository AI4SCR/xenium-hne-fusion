import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.data import compute_tile_stats as module
from xenium_hne_fusion.pipeline import compute_tile_stats_from_items
from xenium_hne_fusion.utils.getters import compute_item_stats


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

    stats = compute_item_stats(
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
    output_dir = tmp_path / '03_output' / 'hest1k'
    tile_dir = tmp_path / 'tiles' / '0'
    tile_dir.mkdir(parents=True)
    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True)
    items_path.write_text(json.dumps([
        {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': 0,
            'tile_dir': str(tile_dir),
        }
    ]))

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

    module.main(dataset='hest1k', items_path=items_path, output_dir=output_dir)

    assert (output_dir / 'statistics' / 'subset.parquet').exists()
    figures_dir = output_dir / 'figures' / 'tile_stats' / 'subset'
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_tile_stats_from_items_accepts_dict_shaped_json(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    tile_dir = tmp_path / 'tiles' / '0'
    tile_dir.mkdir(parents=True)

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

    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps({
        'row0': {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': 0,
            'tile_dir': str(tile_dir),
        }
    }))

    stats_path = compute_tile_stats_from_items(items_path, output_dir)

    assert stats_path == output_dir / 'statistics' / 'subset.parquet'
    assert stats_path.exists()
    figures_dir = output_dir / 'figures' / 'tile_stats' / 'subset'
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_tile_stats_from_items_writes_markdown_summary(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    s1_tile_dir = tmp_path / 'tiles' / 'S1' / '0'
    s2_tile_dir = tmp_path / 'tiles' / 'S2' / '0'
    s1_tile_dir.mkdir(parents=True)
    s2_tile_dir.mkdir(parents=True)

    s1_transcripts = gpd.GeoDataFrame(
        {
            'feature_name': pd.Categorical(
                ['A', 'A', 'B'],
                categories=['A', 'B', 'C'],
                ordered=False,
            )
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    s2_transcripts = gpd.GeoDataFrame(
        {
            'feature_name': pd.Categorical(
                ['B', 'C', 'C'],
                categories=['B', 'C', 'D'],
                ordered=False,
            )
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    cells = gpd.GeoDataFrame(
        {'Level3_grouped': ['tumor', 'stroma', 'tumor']},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    s1_transcripts.to_parquet(s1_tile_dir / 'transcripts.parquet')
    s2_transcripts.to_parquet(s2_tile_dir / 'transcripts.parquet')
    cells.to_parquet(s1_tile_dir / 'cells.parquet')
    cells.to_parquet(s2_tile_dir / 'cells.parquet')

    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps([
        {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': 0,
            'tile_dir': str(s1_tile_dir),
        },
        {
            'id': 'S2_0',
            'sample_id': 'S2',
            'tile_id': 0,
            'tile_dir': str(s2_tile_dir),
        },
    ]))

    compute_tile_stats_from_items(items_path, output_dir)

    summary_path = output_dir / 'statistics' / 'subset.md'
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert 'num_samples' in summary
    assert 'num_transcripts' in summary
    assert 'gene_panel_min' in summary
    assert 'gene_panel_max' in summary
    assert 'gene_panel_intersection' in summary
    assert 'gene_panel_union' in summary
    assert '- `gene_panel_intersection`: 2' in summary
    assert '- `gene_panel_union`: 4' in summary
