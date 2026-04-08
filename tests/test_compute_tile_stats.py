import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.data import compute_tile_stats as module
from xenium_hne_fusion.pipeline import compute_tile_stats_from_items
from xenium_hne_fusion.utils.getters import compute_item_stats, load_processing_config


def _write_expr_tile_inputs(
    tile_dir: Path,
    *,
    transcript_features: list[str],
    expr: pd.DataFrame,
    cell_types: list[str] | None = None,
    feature_universe: list[str] | None = None,
) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = tile_dir.parent.parent
    feature_universe = feature_universe or sorted(set(transcript_features))
    (sample_dir / "feature_universe.txt").write_text("\n".join(feature_universe) + "\n")

    transcripts = gpd.GeoDataFrame(
        {"feature_name": transcript_features},
        geometry=[Point(i, i) for i in range(len(transcript_features))],
    )
    transcripts.to_parquet(tile_dir / "transcripts.parquet")
    expr.to_parquet(tile_dir / "expr-kernel_size=16.parquet")

    if cell_types is not None:
        cells = gpd.GeoDataFrame(
            {"Level3_grouped": cell_types},
            geometry=[Point(i, i) for i in range(len(cell_types))],
        )
        cells.to_parquet(tile_dir / "cells.parquet")


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


def test_main_uses_configured_items_path(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / 'data'
    output_dir = data_dir / '03_output' / 'hest1k'
    tile_dir = tmp_path / 'tiles' / 'S1' / '256_256' / '0'
    config_path = tmp_path / 'hest1k.yaml'
    items_path = output_dir / 'items' / 'subset.json'
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(tmp_path / 'raw' / 'hest1k'))
    items_path.parent.mkdir(parents=True)
    items_path.write_text(json.dumps([
        {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': 0,
            'tile_dir': str(tile_dir),
        }
    ]))

    _write_expr_tile_inputs(
        tile_dir,
        transcript_features=['A', 'A', 'B'],
        expr=pd.DataFrame(
            {
                'A': [1, 1, 0],
                'B': [0, 0, 1],
                'C': [0, 0, 0],
            },
            index=pd.Index([0, 1, 2], name='token_index'),
        ),
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['A', 'B', 'C'],
    )

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 512\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'items:\n'
        '  name: subset\n'
        'split:\n'
        '  name: default\n'
        'panel:\n'
        '  name: default\n'
        '  n_top_genes: null\n'
        '  flavor: null\n'
    )
    processing_cfg = load_processing_config(config_path)

    module.main(processing_cfg, overwrite=True, batch_size=2, num_workers=0)

    assert (output_dir / 'statistics' / 'subset.parquet').exists()
    figures_dir = output_dir / 'figures' / 'tile_stats' / 'subset'
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_tile_stats_from_items_batches_transcript_targets(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    tile_dir = tmp_path / 'tiles' / 'S1' / '256_256' / '0'

    _write_expr_tile_inputs(
        tile_dir,
        transcript_features=['A', 'A', 'B'],
        expr=pd.DataFrame(
            {
                'A': [1, 1, 0],
                'B': [0, 0, 1],
                'C': [0, 0, 0],
            },
            index=pd.Index([0, 1, 2], name='token_index'),
        ),
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['A', 'B', 'C'],
    )

    items_path = output_dir / 'items' / 'subset.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps([
        {
            'id': 'S1_0',
            'sample_id': 'S1',
            'tile_id': 0,
            'tile_dir': str(tile_dir),
        }
    ]))

    stats_path = compute_tile_stats_from_items(items_path, output_dir, batch_size=2, num_workers=0)

    assert stats_path == output_dir / 'statistics' / 'subset.parquet'
    assert stats_path.exists()
    stats = pd.read_parquet(stats_path)
    assert stats.loc['S1_0', 'num_transcripts'] == 3
    assert stats.loc['S1_0', 'num_unique_transcripts'] == 2
    assert pd.isna(stats.loc['S1_0', 'num_cells'])
    assert pd.isna(stats.loc['S1_0', 'num_unique_cells'])
    figures_dir = output_dir / 'figures' / 'tile_stats' / 'subset'
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_linear.png').exists()
    assert (figures_dir / 'num_transcripts_vs_num_unique_transcripts_log.png').exists()


def test_compute_tile_stats_from_items_writes_markdown_summary(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    s1_tile_dir = tmp_path / 'tiles' / 'S1' / '256_256' / '0'
    s2_tile_dir = tmp_path / 'tiles' / 'S2' / '256_256' / '0'

    _write_expr_tile_inputs(
        s1_tile_dir,
        transcript_features=['A', 'A', 'B'],
        expr=pd.DataFrame(
            {
                'A': [1, 1, 0],
                'B': [0, 0, 1],
                'C': [0, 0, 0],
            },
            index=pd.Index([0, 1, 2], name='token_index'),
        ),
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['A', 'B', 'C'],
    )
    _write_expr_tile_inputs(
        s2_tile_dir,
        transcript_features=['B', 'C', 'C'],
        expr=pd.DataFrame(
            {
                'B': [1, 0, 0],
                'C': [0, 1, 1],
                'D': [0, 0, 0],
            },
            index=pd.Index([0, 1, 2], name='token_index'),
        ),
        cell_types=['tumor', 'stroma', 'tumor'],
        feature_universe=['B', 'C', 'D'],
    )

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

    compute_tile_stats_from_items(items_path, output_dir, batch_size=2, num_workers=0)

    summary_path = output_dir / 'statistics' / 'subset.md'
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert 'num_samples' in summary
    assert 'num_transcripts' in summary
    assert 'gene_panel_min' in summary
    assert 'gene_panel_max' in summary
    assert 'gene_panel_intersection' in summary
    assert 'gene_panel_union' in summary
    assert '- `gene_panel_intersection`: 1' in summary
    assert '- `gene_panel_union`: 3' in summary
