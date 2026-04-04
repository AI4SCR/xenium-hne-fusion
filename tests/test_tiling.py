from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xenium_hne_fusion.tiling import save_transcript_overview


def test_save_transcript_overview_streams_row_groups(monkeypatch, tmp_path: Path):
    calls: list[int] = []

    class FakeMetadata:
        num_row_groups = 3
        num_rows = 30

    class FakeParquetFile:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = FakeMetadata()

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
