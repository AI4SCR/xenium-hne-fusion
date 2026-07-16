from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.download import create_structured_symlinks, get_hest_sample_mpp, validate_hest_sample_mpp


def test_create_structured_symlinks_creates_sample_symlinks(tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    structured_dir = tmp_path / 'structured'
    wsi_path = raw_dir / 'wsis' / 'TENX95.tif'
    transcripts_path = raw_dir / 'transcripts' / 'TENX95_transcripts.parquet'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    transcripts_path.write_text('transcripts')

    create_structured_symlinks('TENX95', raw_dir, structured_dir)

    sample_dir = structured_dir / 'TENX95'
    assert (sample_dir / 'wsi.tiff').is_symlink()
    assert (sample_dir / 'transcripts.parquet').is_symlink()


def test_validate_hest_sample_mpp_is_silent_when_within_tolerance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    wsi_path = raw_dir / 'wsis' / 'NCBI783.tif'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    pd.DataFrame(
        [
            {
                'id': 'NCBI783',
                'pixel_size_um_estimated': 0.27396,
            }
        ]
    ).to_csv(metadata_path, index=False)

    class FakeWSI:
        class Properties:
            mpp = 0.27400

        properties = Properties()

    warnings = []
    monkeypatch.setattr('xenium_hne_fusion.download.open_wsi', lambda path: FakeWSI())
    monkeypatch.setattr('xenium_hne_fusion.download.logger.warning', warnings.append)

    validate_hest_sample_mpp('NCBI783', raw_dir, metadata_path)

    assert warnings == []


def test_validate_hest_sample_mpp_warns_on_large_relative_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    wsi_path = raw_dir / 'wsis' / 'NCBI856.tif'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    pd.DataFrame(
        [
            {
                'id': 'NCBI856',
                'pixel_size_um_estimated': 0.2125,
            }
        ]
    ).to_csv(metadata_path, index=False)

    class FakeWSI:
        class Properties:
            mpp = 0.30

        properties = Properties()

    warnings = []
    monkeypatch.setattr('xenium_hne_fusion.download.open_wsi', lambda path: FakeWSI())
    monkeypatch.setattr('xenium_hne_fusion.download.logger.warning', warnings.append)

    validate_hest_sample_mpp('NCBI856', raw_dir, metadata_path)

    assert len(warnings) == 1
    assert 'HEST MPP mismatch for NCBI856' in warnings[0]
    assert 'relative_error=' in warnings[0]


def test_validate_hest_sample_mpp_warns_when_wsi_mpp_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    wsi_path = raw_dir / 'wsis' / 'TENX116.tif'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    pd.DataFrame(
        [
            {
                'id': 'TENX116',
                'pixel_size_um_estimated': 0.136887,
            }
        ]
    ).to_csv(metadata_path, index=False)

    class FakeWSI:
        class Properties:
            mpp = None

        properties = Properties()

    warnings = []
    monkeypatch.setattr('xenium_hne_fusion.download.open_wsi', lambda path: FakeWSI())
    monkeypatch.setattr('xenium_hne_fusion.download.logger.warning', warnings.append)

    validate_hest_sample_mpp('TENX116', raw_dir, metadata_path)

    assert len(warnings) == 1
    assert 'WSI has no mpp metadata' in warnings[0]


def test_validate_hest_sample_mpp_warns_when_metadata_row_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    wsi_path = raw_dir / 'wsis' / 'TENX116.tif'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    pd.DataFrame(
        [
            {
                'id': 'OTHER',
                'pixel_size_um_estimated': 0.136887,
            }
        ]
    ).to_csv(metadata_path, index=False)

    warnings = []
    monkeypatch.setattr('xenium_hne_fusion.download.logger.warning', warnings.append)

    validate_hest_sample_mpp('TENX116', raw_dir, metadata_path)

    assert len(warnings) == 1
    assert 'expected 1 metadata row, found 0' in warnings[0]


def test_get_hest_sample_mpp_reads_estimated_value(tmp_path: Path):
    metadata_path = tmp_path / 'HEST_v1_3_0.csv'
    pd.DataFrame(
        [
            {
                'id': 'NCBI783',
                'pixel_size_um_estimated': 0.27396,
            }
        ]
    ).to_csv(metadata_path, index=False)

    assert get_hest_sample_mpp('NCBI783', metadata_path) == pytest.approx(0.27396)
