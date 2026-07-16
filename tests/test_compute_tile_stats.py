import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from xenium_hne_fusion.artifacts.items import compute_item_stats
from xenium_hne_fusion.artifacts.stats import compute_items_stats, plot_items_stats
from xenium_hne_fusion.utils.getters import ManagedPaths


def _write_tile_inputs(
    tile_dir: Path,
    *,
    transcript_features: list[str] | None = None,
    cell_types: list[str] | None = None,
    feature_universe: list[str] | None = None,
) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = tile_dir.parent.parent
    feature_universe = feature_universe or sorted(set(transcript_features or []))
    (sample_dir / 'feature_universe.txt').write_text('\n'.join(feature_universe) + '\n')

    if transcript_features is not None:
        transcripts = gpd.GeoDataFrame(
            {'feature_name': transcript_features},
            geometry=[Point(i, i) for i in range(len(transcript_features))],
        )
        transcripts.to_parquet(tile_dir / 'transcripts.parquet')

    if cell_types is not None:
        cells = gpd.GeoDataFrame(
            {'Level3_grouped': pd.Categorical(cell_types)},
            geometry=[Point(i, i) for i in range(len(cell_types))],
        )
        cells.to_parquet(tile_dir / 'cells.parquet')


def test_compute_item_stats_reads_tile_local_cells_parquet(tmp_path: Path):
    tile_dir = tmp_path / 'S1' / '256_256' / '0'
    _write_tile_inputs(
        tile_dir,
        transcript_features=['A', 'A', 'B'],
        cell_types=['tumor', 'stroma', 'tumor'],
    )

    stats = compute_item_stats({'id': 'S1_0', 'tile_dir': str(tile_dir)}, cell_type_col='Level3_grouped')

    assert stats == {
        'id': 'S1_0',
        'num_transcripts': 3,
        'num_unique_transcripts': 2,
        'num_cells': 3,
        'num_unique_cells': 2,
    }


def test_compute_item_stats_treats_missing_files_as_zero(tmp_path: Path):
    tile_dir = tmp_path / 'S1' / '256_256' / '0'
    _write_tile_inputs(tile_dir, transcript_features=None, cell_types=None, feature_universe=['A'])

    stats = compute_item_stats({'id': 'S1_0', 'tile_dir': str(tile_dir)}, cell_type_col='Level3_grouped')

    assert stats == {
        'id': 'S1_0',
        'num_transcripts': 0,
        'num_unique_transcripts': 0,
        'num_cells': 0,
        'num_unique_cells': 0,
    }


def test_plot_items_stats_writes_transcript_scatter_plots(tmp_path: Path):
    stats = pd.DataFrame(
        {
            'num_transcripts': [10, 100, 1000],
            'num_unique_transcripts': [5, 20, 100],
            'num_cells': [1, 2, 3],
            'num_unique_cells': [1, 2, 2],
        },
        index=['a', 'b', 'c'],
    )

    plot_items_stats(stats, tmp_path)

    assert (tmp_path / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (tmp_path / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_items_stats_includes_tiles_without_transcripts_or_cells(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    with_data = output_dir / 'tiles' / 'S1' / '256_256' / '0'
    empty = output_dir / 'tiles' / 'S1' / '256_256' / '1'

    _write_tile_inputs(
        with_data,
        transcript_features=['A', 'A', 'B'],
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['A', 'B'],
    )
    _write_tile_inputs(empty, transcript_features=None, cell_types=None, feature_universe=['A', 'B'])

    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps([
        {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(with_data)},
        {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': str(empty)},
    ]))

    managed_paths = ManagedPaths(data_dir=tmp_path, name='hest1k')
    stats_path = compute_items_stats(items_path, managed_paths, cell_type_col='Level3_grouped', num_workers=0)
    figures_dir = output_dir / 'figures' / 'items' / 'stats' / 'subset'

    stats = pd.read_parquet(stats_path)
    assert list(stats.columns) == ['num_transcripts', 'num_unique_transcripts', 'num_cells', 'num_unique_cells']
    assert stats.loc['S1_0', 'num_transcripts'] == 3
    assert stats.loc['S1_0', 'num_unique_transcripts'] == 2
    assert stats.loc['S1_0', 'num_cells'] == 3
    assert stats.loc['S1_0', 'num_unique_cells'] == 2
    assert stats.loc['S1_1', 'num_transcripts'] == 0
    assert stats.loc['S1_1', 'num_unique_transcripts'] == 0
    assert stats.loc['S1_1', 'num_cells'] == 0
    assert stats.loc['S1_1', 'num_unique_cells'] == 0

    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_items_stats_writes_markdown_summary(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    s1_tile_dir = output_dir / 'tiles' / 'S1' / '256_256' / '0'
    s2_tile_dir = output_dir / 'tiles' / 'S2' / '256_256' / '0'

    _write_tile_inputs(
        s1_tile_dir,
        transcript_features=['A', 'A', 'B'],
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['A', 'B', 'C'],
    )
    _write_tile_inputs(
        s2_tile_dir,
        transcript_features=['B', 'C', 'C'],
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['B', 'C', 'D'],
    )

    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps([
        {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(s1_tile_dir)},
        {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': str(s2_tile_dir)},
    ]))

    managed_paths = ManagedPaths(data_dir=tmp_path, name='hest1k')
    compute_items_stats(items_path, managed_paths, cell_type_col='Level3_grouped', num_workers=0)

    summary_path = output_dir / 'statistics' / 'subset.md'
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert 'num_samples' in summary
    assert 'num_transcripts' in summary
    assert 'num_transcripts_min' in summary
    assert 'num_transcripts_median' in summary
    assert 'num_transcripts_max' in summary
    assert 'num_cells' in summary
    assert 'num_unique_cells_min' in summary
    assert 'num_unique_cells_median' in summary
    assert 'num_unique_cells_max' in summary
    assert 'gene_panel_min' in summary
    assert 'gene_panel_max' in summary
    assert 'gene_panel_intersection' in summary
    assert 'gene_panel_union' in summary
    assert '- `num_cells`: 6' in summary
    assert '- `num_transcripts_min`: 3' in summary
    assert '- `num_transcripts_median`: 3.0' in summary
    assert '- `num_transcripts_max`: 3' in summary
    assert '- `num_unique_cells_min`: 2' in summary
    assert '- `num_unique_cells_median`: 2.0' in summary
    assert '- `num_unique_cells_max`: 2' in summary
    assert '- `gene_panel_intersection`: 2' in summary
    assert '- `gene_panel_union`: 4' in summary
