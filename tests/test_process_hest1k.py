import importlib.util
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_process_hest1k_uses_metadata_mpp_for_tiling_and_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    structured_dir = data_dir / '01_structured' / 'hest1k'
    sample_dir = structured_dir / 'NCBI856'
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  sample_ids:\n'
        '    - NCBI856\n'
    )

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / 'wsi.tiff').write_text('wsi')
    (sample_dir / 'transcripts.parquet').write_text('transcripts')
    (structured_dir / 'metadata.csv').write_text(
        'id,pixel_size_um_embedded,pixel_size_um_estimated\n'
        'NCBI856,0.2125,0.2125\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/process_hest1k.py', 'process_hest1k_script')

    calls = []

    monkeypatch.setattr(module, 'detect_tissues', lambda wsi_path, output_parquet: calls.append(('detect_tissues', wsi_path, output_parquet)))
    monkeypatch.setattr(
        module,
        'tile_tissues',
        lambda wsi_path, tissues_parquet, tile_px, stride_px, mpp, output_parquet, slide_mpp=None: calls.append(
            ('tile_tissues', wsi_path, tissues_parquet, tile_px, stride_px, mpp, output_parquet, slide_mpp)
        ),
    )
    monkeypatch.setattr(module.gpd, 'read_parquet', lambda path: gpd.GeoDataFrame({'tile_id': [0]}, geometry=[box(0, 0, 4, 4)]))
    monkeypatch.setattr(
        module,
        'extract_tiles',
        lambda wsi_path, tiles, output_dir, mpp, native_mpp=None: calls.append(
            ('extract_tiles', wsi_path, len(tiles), output_dir, mpp, native_mpp)
        ),
    )
    monkeypatch.setattr(
        module,
        'tile_transcripts',
        lambda tiles, transcripts_path, save_dir, predicate='within': calls.append(
            ('tile_transcripts', len(tiles), transcripts_path, save_dir, predicate)
        ),
    )
    monkeypatch.setattr(
        module,
        'process_tiles',
        lambda tiles, transcripts_dir, output_dir, raw_transcripts_path, img_size=256, kernel_size=16: calls.append(
            ('process_tiles', len(tiles), transcripts_dir, output_dir, raw_transcripts_path, img_size, kernel_size)
        ),
    )

    module.main('hest1k', config_path=config_path, sample_id='NCBI856')

    processed_dir = data_dir / '02_processed' / 'hest1k' / 'NCBI856' / '256_256'
    assert calls == [
        ('detect_tissues', sample_dir / 'wsi.tiff', sample_dir / 'tissues.parquet'),
        (
            'tile_tissues',
            sample_dir / 'wsi.tiff',
            sample_dir / 'tissues.parquet',
            256,
            256,
            0.5,
            sample_dir / 'tiles' / '256_256.parquet',
            0.2125,
        ),
        ('extract_tiles', sample_dir / 'wsi.tiff', 1, processed_dir, 0.5, 0.2125),
        ('tile_transcripts', 1, sample_dir / 'transcripts.parquet', processed_dir / 'transcripts', 'within'),
        ('process_tiles', 1, processed_dir / 'transcripts', processed_dir, sample_dir / 'transcripts.parquet', 256, 16),
    ]
