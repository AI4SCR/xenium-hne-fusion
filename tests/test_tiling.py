from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, box

from xenium_hne_fusion.tiling import save_points_overview, save_sample_overview, save_transcript_overview, tile_tissues


def test_save_transcript_overview_streams_batches(monkeypatch, tmp_path: Path):
    calls: list[int] = []
    visualize_calls: list[gpd.GeoDataFrame] = []

    class FakeMetadata:
        num_rows = 3 * 65_536

    class FakeSchema:
        names = ['he_x', 'he_y']

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()
            self.schema_arrow = FakeSchema()

        def read(self, *args, **kwargs):
            raise AssertionError('save_transcript_overview should not load the full parquet file')

        def iter_batches(self, batch_size: int, columns: list[str]):
            assert columns == ['he_x', 'he_y']
            assert batch_size == 65_536
            for index in range(3):
                calls.append(index)
                data = {
                    'he_x': np.arange(index * 10, index * 10 + 10),
                    'he_y': np.arange(index * 10, index * 10 + 10),
                }
                yield pa.table(data)

        def read_row_group(self, index: int, columns: list[str]):
            calls.append(index)
            raise AssertionError('save_transcript_overview should stream batches, not row groups')

    class FakeSlide:
        level_dimensions = [(200, 100)]

        def get_thumbnail(self, size):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def close(self):
            return None

    def fake_visualize_points(points, *, slide=None, image=None, **kwargs):
        assert image is None
        assert slide is not None
        visualize_calls.append(points.copy())
        return np.zeros((20, 40, 3), dtype=np.uint8)

    monkeypatch.setattr(pq, 'ParquetFile', FakeParquetFile)
    monkeypatch.setattr('openslide.OpenSlide', lambda _: FakeSlide())
    monkeypatch.setattr('ai4bmr_learn.plotting.xenium.visualize_points', fake_visualize_points)

    output_path = tmp_path / 'transcripts.png'
    save_transcript_overview(tmp_path / 'wsi.tiff', tmp_path / 'transcripts.parquet', output_path, n=5)

    assert output_path.exists()
    assert calls == [0, 1, 2]
    assert len(visualize_calls) == 1
    assert visualize_calls[0].geometry.x.notna().all()
    assert visualize_calls[0].geometry.y.notna().all()


def test_save_transcript_overview_supports_geometry_only_parquet(monkeypatch, tmp_path: Path):
    from_arrow_calls = []

    class FakeMetadata:
        num_row_groups = 1
        num_rows = 2

    class FakeSchema:
        names = ['geometry']

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()
            self.schema_arrow = FakeSchema()

        def read(self, *args, **kwargs):
            raise AssertionError('save_transcript_overview should not load the full parquet file')

        def iter_batches(self, batch_size: int, columns: list[str]):
            assert columns == ['geometry']
            assert batch_size == 65_536
            yield pa.table({'geometry': [b'a', b'b']})

    class FakeSlide:
        level_dimensions = [(200, 100)]

        def get_thumbnail(self, size):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def close(self):
            return None

    def fake_from_arrow(batch):
        from_arrow_calls.append(batch)
        return gpd.GeoDataFrame({'geometry': [Point(1, 2), Point(3, 4)]}, geometry='geometry')

    def fake_visualize_points(points, *, slide=None, image=None, **kwargs):
        assert list(points.geometry.x) == [1.0, 3.0]
        assert list(points.geometry.y) == [2.0, 4.0]
        return np.zeros((20, 40, 3), dtype=np.uint8)

    monkeypatch.setattr(pq, 'ParquetFile', FakeParquetFile)
    monkeypatch.setattr('openslide.OpenSlide', lambda _: FakeSlide())
    monkeypatch.setattr('xenium_hne_fusion.tiling.gpd.GeoDataFrame.from_arrow', fake_from_arrow)
    monkeypatch.setattr('ai4bmr_learn.plotting.xenium.visualize_points', fake_visualize_points)

    output_path = tmp_path / 'transcripts.png'
    save_transcript_overview(tmp_path / 'wsi.tiff', tmp_path / 'transcripts.parquet', output_path, n=2)

    assert output_path.exists()
    assert len(from_arrow_calls) == 1


def test_save_transcript_overview_requires_coordinates(monkeypatch, tmp_path: Path):
    class FakeMetadata:
        num_row_groups = 1
        num_rows = 2

    class FakeSchema:
        names = ['feature_name']

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()
            self.schema_arrow = FakeSchema()

    monkeypatch.setattr(pq, 'ParquetFile', FakeParquetFile)

    with pytest.raises(AssertionError, match='Missing point coordinates'):
        save_transcript_overview(tmp_path / 'wsi.tiff', tmp_path / 'transcripts.parquet', tmp_path / 'transcripts.png', n=2)


def test_save_points_overview_supports_cells_geometry(monkeypatch, tmp_path: Path):
    class FakeMetadata:
        num_rows = 2

    class FakeSchema:
        names = ['geometry']

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()
            self.schema_arrow = FakeSchema()

        def iter_batches(self, batch_size: int, columns: list[str]):
            assert columns == ['geometry']
            yield pa.table({'geometry': [b'a', b'b']})

    class FakeSlide:
        def close(self):
            return None

    def fake_from_arrow(batch):
        return gpd.GeoDataFrame({'geometry': [Point(1, 2), Point(3, 4)]}, geometry='geometry')

    def fake_visualize_points(points, *, slide=None, image=None, **kwargs):
        assert list(points.geometry.x) == [1.0, 3.0]
        assert list(points.geometry.y) == [2.0, 4.0]
        return np.zeros((20, 40, 3), dtype=np.uint8)

    monkeypatch.setattr(pq, 'ParquetFile', FakeParquetFile)
    monkeypatch.setattr('openslide.OpenSlide', lambda _: FakeSlide())
    monkeypatch.setattr('xenium_hne_fusion.tiling.gpd.GeoDataFrame.from_arrow', fake_from_arrow)
    monkeypatch.setattr('ai4bmr_learn.plotting.xenium.visualize_points', fake_visualize_points)

    output_path = tmp_path / 'cells.png'
    save_points_overview(tmp_path / 'wsi.tiff', tmp_path / 'cells.parquet', output_path, n=2, label='cells')

    assert output_path.exists()


def test_save_sample_overview_uses_points_stem_for_output(monkeypatch, tmp_path: Path):
    calls = []
    wsi_path = tmp_path / 'wsi.tiff'
    transcripts_path = tmp_path / 'transcripts.parquet'
    cells_path = tmp_path / 'cells.parquet'

    monkeypatch.setattr('xenium_hne_fusion.tiling.save_wsi_thumbnail', lambda *args, **kwargs: calls.append(('wsi', args)))
    monkeypatch.setattr(
        'xenium_hne_fusion.tiling.save_points_overview',
        lambda wsi_path, points_path, output_path, n, max_size, label=None: calls.append((points_path, output_path)),
    )

    save_sample_overview(wsi_path, transcripts_path, tmp_path)
    save_sample_overview(wsi_path, cells_path, tmp_path)

    assert (transcripts_path, tmp_path / 'transcripts.png') in calls
    assert (cells_path, tmp_path / 'cells.png') in calls


def test_tile_tissues_passes_slide_mpp_override(monkeypatch, tmp_path: Path):
    calls = []

    class FakeWsi:
        def __init__(self):
            self.data = {}

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data[key]

    fake_wsi = FakeWsi()

    monkeypatch.setattr('xenium_hne_fusion.tiling.open_wsi', lambda _: fake_wsi)
    monkeypatch.setattr('xenium_hne_fusion.tiling.ShapesModel.parse', lambda df: df)
    monkeypatch.setattr(
        'xenium_hne_fusion.tiling.gpd.read_parquet',
        lambda _: gpd.GeoDataFrame({'tissue_id': [0]}, geometry=[box(0, 0, 10, 10)]),
    )

    def fake_tile_tissues(wsi, tile_px, stride_px, mpp, slide_mpp):
        calls.append((tile_px, stride_px, mpp, slide_mpp))
        wsi['tiles'] = gpd.GeoDataFrame({'tile_id': [0], 'tissue_id': [0]}, geometry=[box(0, 0, 4, 4)])

    monkeypatch.setattr('xenium_hne_fusion.tiling.zs.pp.tile_tissues', fake_tile_tissues)

    output_path = tmp_path / 'tiles.parquet'
    tile_tissues(
        tmp_path / 'wsi.tiff',
        tmp_path / 'tissues.parquet',
        tile_px=256,
        stride_px=256,
        mpp=0.5,
        output_parquet=output_path,
        slide_mpp=0.2125,
    )

    assert calls == [(256, 256, 0.5, 0.2125)]
    assert output_path.exists()
