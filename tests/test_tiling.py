from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box

from xenium_hne_fusion.tiling import save_transcript_overview, tile_tissues


def test_save_transcript_overview_streams_row_groups(monkeypatch, tmp_path: Path):
    calls: list[int] = []

    class FakeMetadata:
        num_row_groups = 3
        num_rows = 30

    class FakeSchema:
        names = ['he_x', 'he_y']

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()
            self.schema_arrow = FakeSchema()

        def read(self, *args, **kwargs):
            raise AssertionError('save_transcript_overview should not load the full parquet file')

        def read_row_group(self, index: int, columns: list[str]):
            calls.append(index)
            data = {
                'he_x': np.arange(index * 10, index * 10 + 10),
                'he_y': np.arange(index * 10, index * 10 + 10),
            }
            return pa.table(data)

    class FakeReaderProperties:
        level_shape = [(100, 200)]

    class FakeReader:
        properties = FakeReaderProperties()

        def get_thumbnail(self, max_size: int):
            return np.zeros((20, 40, 3), dtype=np.uint8)

    class FakeWsi:
        reader = FakeReader()

    monkeypatch.setattr(pq, 'ParquetFile', FakeParquetFile)
    monkeypatch.setattr('xenium_hne_fusion.tiling.open_wsi', lambda _: FakeWsi())

    output_path = tmp_path / 'transcripts.png'
    save_transcript_overview(tmp_path / 'wsi.tiff', tmp_path / 'transcripts.parquet', output_path, n=5)

    assert output_path.exists()
    assert calls


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
