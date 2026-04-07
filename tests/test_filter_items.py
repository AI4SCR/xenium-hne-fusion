import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_filter_items_module():
    path = Path('scripts/data/filter_items.py').resolve()
    spec = importlib.util.spec_from_file_location('filter_items_script', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_items_uses_beat_default_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'beat'
    output_dir = data_dir / '03_output' / 'beat'
    config_path = tmp_path / 'beat.yaml'

    config_path.write_text(
        'name: beat\n'
        'tile_px: 512\n'
        'stride_px: 512\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  sample_ids: null\n'
    )
    items_config_path = tmp_path / 'beat-items.yaml'
    items_config_path.write_text(
        'name: default\n'
        'num_transcripts: 200\n'
    )

    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'items' / 'all.json').write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/S1/0'},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/S1/1'},
                {'id': 'S1_2', 'sample_id': 'S1', 'tile_id': 2, 'tile_dir': '/tmp/S1/2'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [199, 200, 400],
            'num_unique_transcripts': [None, None, None],
            'num_cells': [None, None, None],
            'num_unique_cells': [None, None, None],
        },
        index=pd.Index(['S1_0', 'S1_1', 'S1_2'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'all.parquet')

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('BEAT_RAW_DIR', str(raw_dir))

    module = _load_filter_items_module()
    module.main(
        'beat',
        items_config_path=items_config_path,
        config_path=config_path,
        overwrite=True,
    )

    filtered = json.loads((output_dir / 'items' / 'default.json').read_text())
    assert [item['id'] for item in filtered] == ['S1_1', 'S1_2']


def test_filter_items_filters_hest1k_by_organ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    output_dir = data_dir / '03_output' / 'hest1k'
    processed_dir = data_dir / '02_processed' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  sample_ids: null\n'
    )

    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / 'items' / 'all.json').write_text(
        json.dumps(
            [
                {'id': 'L1_0', 'sample_id': 'L1', 'tile_id': 0, 'tile_dir': '/tmp/L1/0'},
                {'id': 'B1_0', 'sample_id': 'B1', 'tile_id': 0, 'tile_dir': '/tmp/B1/0'},
                {'id': 'L1_1', 'sample_id': 'L1', 'tile_id': 1, 'tile_dir': '/tmp/L1/1'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [100, 1000, 99],
            'num_unique_transcripts': [None, None, None],
            'num_cells': [None, None, None],
            'num_unique_cells': [None, None, None],
        },
        index=pd.Index(['L1_0', 'B1_0', 'L1_1'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'all.parquet')
    pd.DataFrame(
        [
            {'sample_id': 'L1', 'organ': 'Lung'},
            {'sample_id': 'B1', 'organ': 'Breast'},
        ]
    ).to_parquet(processed_dir / 'metadata.parquet', index=False)

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_filter_items_module()
    module.main(
        'hest1k',
        items_config_path=Path('configs/items/hest1k/lung.yaml'),
        config_path=config_path,
        overwrite=True,
    )

    filtered = json.loads((output_dir / 'items' / 'lung.json').read_text())
    assert [item['id'] for item in filtered] == ['L1_0']
