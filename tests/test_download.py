import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.download import create_structured_symlinks, get_hest_sample_mpp, validate_hest_sample_mpp
from xenium_hne_fusion.utils.getters import load_processing_config


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_structured_symlinks_creates_sample_visualizations(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    structured_dir = tmp_path / 'structured'
    wsi_path = raw_dir / 'wsis' / 'TENX95.tif'
    transcripts_path = raw_dir / 'transcripts' / 'TENX95_transcripts.parquet'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    transcripts_path.write_text('transcripts')

    calls = []

    def fake_save_sample_overview(wsi_path: Path, transcripts_path: Path, output_dir: Path, n: int = 10_000, max_size: int = 2048):
        calls.append((wsi_path, transcripts_path, output_dir, n, max_size))

    monkeypatch.setattr('xenium_hne_fusion.structure.save_sample_overview', fake_save_sample_overview)

    create_structured_symlinks('TENX95', raw_dir, structured_dir)

    sample_dir = structured_dir / 'TENX95'
    assert (sample_dir / 'wsi.tiff').is_symlink()
    assert (sample_dir / 'transcripts.parquet').is_symlink()
    assert calls == [(sample_dir / 'wsi.tiff', sample_dir / 'transcripts.parquet', sample_dir, 10_000, 2048)]


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


def test_structure_hest1k_validates_mpp_after_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'img_size: 224\n'
        'filter:\n'
        '  include_ids:\n'
        '    - NCBI783\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/structure_hest1k.py', 'structure_hest1k_script')

    calls = []
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'

    def fake_download_hest_metadata(raw_dir_arg: Path) -> Path:
        calls.append(('download_hest_metadata', raw_dir_arg))
        return metadata_path

    def fake_create_structured_metadata_symlink(metadata_path_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(('create_structured_metadata_symlink', metadata_path_arg, structured_dir_arg))

    def fake_download_sample(sample_id: str, raw_dir_arg: Path) -> Path:
        calls.append(('download_sample', sample_id, raw_dir_arg))
        return raw_dir_arg

    def fake_validate_hest_sample_mpp(sample_id: str, raw_dir_arg: Path, metadata_path_arg: Path) -> None:
        calls.append(('validate_hest_sample_mpp', sample_id, raw_dir_arg, metadata_path_arg))

    def fake_create_structured_symlinks(sample_id: str, raw_dir_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(('create_structured_symlinks', sample_id, raw_dir_arg, structured_dir_arg))

    monkeypatch.setattr(module, 'download_hest_metadata', fake_download_hest_metadata)
    monkeypatch.setattr(module, 'create_structured_metadata_symlink', fake_create_structured_metadata_symlink)
    monkeypatch.setattr(module, 'download_sample', fake_download_sample)
    monkeypatch.setattr(module, 'validate_hest_sample_mpp', fake_validate_hest_sample_mpp)
    monkeypatch.setattr(module, 'create_structured_symlinks', fake_create_structured_symlinks)
    monkeypatch.setattr(module, 'resolve_samples', lambda cfg, metadata_csv: ['NCBI783'])

    module.main(load_processing_config(config_path))

    structured_dir = data_dir / '01_structured' / 'hest1k'
    assert calls == [
        ('download_hest_metadata', raw_dir.resolve()),
        ('create_structured_metadata_symlink', metadata_path, structured_dir.resolve()),
        ('download_sample', 'NCBI783', raw_dir.resolve()),
        ('validate_hest_sample_mpp', 'NCBI783', raw_dir.resolve(), metadata_path),
        ('create_structured_symlinks', 'NCBI783', raw_dir.resolve(), structured_dir.resolve()),
    ]


def test_structure_hest1k_reuses_existing_raw_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'img_size: 224\n'
        'filter:\n'
        '  include_ids:\n'
        '    - NCBI783\n'
    )

    (raw_dir / 'wsis').mkdir(parents=True, exist_ok=True)
    (raw_dir / 'transcripts').mkdir(parents=True, exist_ok=True)
    metadata_path = raw_dir / 'HEST_v1_3_0.csv'
    metadata_path.write_text('id\nNCBI783\n')
    (raw_dir / 'wsis' / 'NCBI783.tif').write_text('wsi')
    (raw_dir / 'transcripts' / 'NCBI783_transcripts.parquet').write_text('tx')

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/structure_hest1k.py', 'structure_hest1k_reuse_script')

    calls = []

    def fail_download_hest_metadata(raw_dir_arg: Path) -> Path:
        raise AssertionError(f'download_hest_metadata should not be called for {raw_dir_arg}')

    def fake_create_structured_metadata_symlink(metadata_path_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(('create_structured_metadata_symlink', metadata_path_arg, structured_dir_arg))

    def fail_download_sample(sample_id: str, raw_dir_arg: Path) -> Path:
        raise AssertionError(f'download_sample should not be called for {sample_id} in {raw_dir_arg}')

    def fake_validate_hest_sample_mpp(sample_id: str, raw_dir_arg: Path, metadata_path_arg: Path) -> None:
        calls.append(('validate_hest_sample_mpp', sample_id, raw_dir_arg, metadata_path_arg))

    def fake_create_structured_symlinks(sample_id: str, raw_dir_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(('create_structured_symlinks', sample_id, raw_dir_arg, structured_dir_arg))

    monkeypatch.setattr(module, 'download_hest_metadata', fail_download_hest_metadata)
    monkeypatch.setattr(module, 'create_structured_metadata_symlink', fake_create_structured_metadata_symlink)
    monkeypatch.setattr(module, 'download_sample', fail_download_sample)
    monkeypatch.setattr(module, 'validate_hest_sample_mpp', fake_validate_hest_sample_mpp)
    monkeypatch.setattr(module, 'create_structured_symlinks', fake_create_structured_symlinks)

    module.main(load_processing_config(config_path))

    structured_dir = data_dir / '01_structured' / 'hest1k'
    assert calls == [
        ('create_structured_metadata_symlink', metadata_path, structured_dir.resolve()),
        ('validate_hest_sample_mpp', 'NCBI783', raw_dir.resolve(), metadata_path),
        ('create_structured_symlinks', 'NCBI783', raw_dir.resolve(), structured_dir.resolve()),
    ]
