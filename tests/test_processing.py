from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from shapely.geometry import Point, box

from xenium_hne_fusion.processing import (
    compute_expr_tokens,
    expr_pool,
    extract_tiles,
    infer_feature_universe,
    make_token_tiles,
    process_cells,
    process_tiles,
    set_feature_universe,
    tile_cells,
    tile_transcripts,
)


def test_infer_feature_universe_streams_and_caches_filtered_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[tuple[int, list[str]]] = []

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path

        def iter_batches(self, batch_size: int, columns: list[str]):
            calls.append((batch_size, columns))
            yield pa.record_batch([pa.array([b"B", b"BLANK_1", b"A"])], names=["feature_name"])
            yield pa.record_batch([pa.array([b"C", b"A", b"NegControlProbe_x"])], names=["feature_name"])

    monkeypatch.setattr("xenium_hne_fusion.processing.pq.ParquetFile", FakeParquetFile)

    transcripts_path = tmp_path / "raw.parquet"
    feature_universe_path = tmp_path / "feature_universe.txt"

    feature_universe = infer_feature_universe(
        transcripts_path,
        feature_universe_path=feature_universe_path,
        batch_size=2,
    )

    assert feature_universe == ["A", "B", "C"]
    assert feature_universe_path.read_text() == "A\nB\nC\n"
    assert calls == [(2, ["feature_name"])]

    class FailingParquetFile:
        def __init__(self, path: Path):
            raise AssertionError("cached feature_universe.txt should be reused")

    monkeypatch.setattr("xenium_hne_fusion.processing.pq.ParquetFile", FailingParquetFile)
    assert infer_feature_universe(transcripts_path, feature_universe_path=feature_universe_path) == ["A", "B", "C"]


def test_expr_pool_row_reindex_does_not_add_missing_feature_columns():
    expr_subsets = pd.DataFrame(
        {
            "token_index": [0, 2],
            "feature_name": pd.Categorical(["A", "C"], categories=["A", "C"], ordered=False),
        }
    )

    tokens = expr_pool(expr_subsets, num_tokens=4, group_by="feature_name")

    assert tokens.index.tolist() == [0, 1, 2, 3]
    assert tokens.columns.tolist() == ["A", "C"]


def test_tile_transcripts_roundtrip_categorical_and_expr_columns_use_full_feature_universe(tmp_path: Path):
    feature_universe = ["A", "B", "C", "D"]
    token_tiles = make_token_tiles(256, 128)

    points = gpd.GeoDataFrame(
        {
            "transcript_id": [1, 2],
            "cell_id": [10, 11],
            "feature_name": set_feature_universe(pd.Series(["A", "C"], name="feature_name"), feature_universe),
        },
        geometry=[Point(10, 10), Point(200, 10)],
    )

    transcripts_path = tmp_path / "transcripts.parquet"
    points.to_parquet(transcripts_path)

    stored = pd.read_parquet(transcripts_path)
    assert isinstance(stored["feature_name"].dtype, pd.CategoricalDtype)
    assert stored["feature_name"].cat.categories.tolist() == feature_universe

    expr = compute_expr_tokens(points, token_tiles, feature_universe=feature_universe)

    assert expr.columns.tolist() == feature_universe
    assert expr.shape == (4, 4)
    assert expr["B"].sum() == 0
    assert expr["D"].sum() == 0
    assert expr.loc[0, "A"] == 1
    assert expr.loc[1, "C"] == 1


def test_processed_sample_expr_columns_match_feature_universe_smoke():
    sample_dir = Path("data/02_processed/hest1k/TENX95")
    expr_root = sample_dir / "256_256"
    feature_universe_path = sample_dir / "feature_universe.txt"
    if not feature_universe_path.exists() or not expr_root.exists():
        pytest.skip("Processed sample with cached feature universe is not available")

    feature_universe = feature_universe_path.read_text().splitlines()
    expr_paths = sorted(expr_root.glob("*/expr-kernel_size=16.parquet"))
    if not expr_paths:
        pytest.skip("No processed expr parquet files found")

    for expr_path in expr_paths:
        schema = pq.read_schema(expr_path)
        columns = [name for name in schema.names if name != "token_index"]
        assert columns == feature_universe

    transcript_path = expr_paths[0].with_name("transcripts.parquet")
    stored = pd.read_parquet(transcript_path)
    assert isinstance(stored["feature_name"].dtype, pd.CategoricalDtype)
    assert stored["feature_name"].cat.categories.tolist() == feature_universe


def test_extract_tiles_uses_native_mpp_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeReader:
        def get_region(self, x, y, w, h, level=0):
            return np.zeros((h, w, 3), dtype=np.uint8)

    class FakeProperties:
        mpp = None

    class FakeWsi:
        reader = FakeReader()
        properties = FakeProperties()

    monkeypatch.setattr('xenium_hne_fusion.processing.open_wsi', lambda _: FakeWsi())

    tiles = gpd.GeoDataFrame(
        [{'tile_id': 0, 'x_px': 0, 'y_px': 0, 'width_px': 100, 'height_px': 100}],
        geometry=[Point(0, 0)],
    )
    output_dir = tmp_path / 'processed'

    extract_tiles(tmp_path / 'wsi.tiff', tiles, output_dir, mpp=0.5, native_mpp=0.25)

    tensor = __import__('torch').load(output_dir / '0' / 'tile.pt', weights_only=True)
    assert tuple(tensor.shape) == (3, 50, 50)


def test_tile_subsets_are_written_directly_into_tile_dirs(tmp_path: Path):
    tiles = gpd.GeoDataFrame(
        [
            {'tile_id': 0, 'x_px': 0, 'y_px': 0, 'width_px': 100, 'height_px': 100},
            {'tile_id': 1, 'x_px': 100, 'y_px': 0, 'width_px': 100, 'height_px': 100},
        ],
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
    )
    transcripts = gpd.GeoDataFrame(
        {
            'transcript_id': [1, 2],
            'cell_id': [10, 11],
            'feature_name': ['A', 'B'],
        },
        geometry=[Point(10, 10), Point(150, 20)],
    )
    cells = gpd.GeoDataFrame(
        {
            'original_cell_id': [100, 101],
            'Level3_grouped': ['tumor', 'stroma'],
        },
        geometry=[Point(20, 20), Point(120, 30)],
    )

    transcripts_path = tmp_path / 'transcripts.parquet'
    cells_path = tmp_path / 'cells.parquet'
    transcripts.to_parquet(transcripts_path)
    cells.to_parquet(cells_path)

    output_dir = tmp_path / 'processed'
    tile_transcripts(tiles, transcripts_path, output_dir)
    tile_cells(tiles, cells_path, output_dir)

    assert not (output_dir / 'transcripts').exists()
    assert not (output_dir / 'cells').exists()
    assert (output_dir / '0' / 'transcripts.parquet').exists()
    assert (output_dir / '1' / 'transcripts.parquet').exists()
    assert (output_dir / '0' / 'cells.parquet').exists()
    assert (output_dir / '1' / 'cells.parquet').exists()

    stored_tx = gpd.read_parquet(output_dir / '0' / 'transcripts.parquet')
    stored_cells = gpd.read_parquet(output_dir / '0' / 'cells.parquet')
    assert 'tile_id' not in stored_tx.columns
    assert 'tile_id' not in stored_cells.columns


def test_process_tiles_and_cells_create_expected_tile_local_artifacts(tmp_path: Path):
    tiles = gpd.GeoDataFrame(
        [{'tile_id': 0, 'x_px': 0, 'y_px': 0, 'width_px': 100, 'height_px': 100}],
        geometry=[box(0, 0, 100, 100)],
    )
    transcripts = gpd.GeoDataFrame(
        {
            'transcript_id': [1, 2],
            'cell_id': [10, 11],
            'feature_name': ['A', 'B'],
        },
        geometry=[Point(10, 10), Point(60, 20)],
    )
    cells = gpd.GeoDataFrame(
        {
            'original_cell_id': [100, 101],
            'Level3_grouped': ['tumor', 'stroma'],
        },
        geometry=[Point(15, 15), Point(75, 35)],
    )

    transcripts_path = tmp_path / 'transcripts.parquet'
    cells_path = tmp_path / 'cells.parquet'
    transcripts.to_parquet(transcripts_path)
    cells.to_parquet(cells_path)

    output_dir = tmp_path / 'processed'
    tile_dir = output_dir / '0'
    tile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros((3, 100, 100), dtype=torch.uint8), tile_dir / 'tile.pt')

    tile_transcripts(tiles, transcripts_path, output_dir)
    tile_cells(tiles, cells_path, output_dir)
    process_tiles(tiles, output_dir, transcripts_path, img_size=100, kernel_size=50)
    process_cells(tiles, output_dir, img_size=100)

    assert (tile_dir / 'transcripts.parquet').exists()
    assert (tile_dir / 'expr-kernel_size=50.parquet').exists()
    assert (tile_dir / 'tile.png').exists()
    assert (tile_dir / 'transcripts.png').exists()
    assert (tile_dir / 'transcripts_top5_feats.png').exists()
    assert (tile_dir / 'cells.parquet').exists()
    assert (tile_dir / 'cells.png').exists()

    stored_tx = gpd.read_parquet(tile_dir / 'transcripts.parquet')
    stored_cells = gpd.read_parquet(tile_dir / 'cells.parquet')
    assert stored_tx.geometry.x.between(0, 100).all()
    assert stored_tx.geometry.y.between(0, 100).all()
    assert stored_cells.geometry.x.between(0, 100).all()
    assert stored_cells.geometry.y.between(0, 100).all()
