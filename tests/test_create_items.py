import importlib.util
import json
from pathlib import Path

import pytest


def _load_create_items_module():
    path = Path('scripts/data/create_items.py').resolve()
    spec = importlib.util.spec_from_file_location('create_items_script', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_items_writes_dataset_scoped_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    processed_dir = data_dir / '02_processed' / 'hest1k' / 'TENX95'
    complete_tile = processed_dir / '0'
    incomplete_tile = processed_dir / '1'
    config_path = tmp_path / 'hest1k.yaml'

    for path in [complete_tile, incomplete_tile]:
        path.mkdir(parents=True, exist_ok=True)
    for filename in ['tile.pt', 'expr-kernel_size=16.parquet', 'transcripts.parquet']:
        (complete_tile / filename).write_text('')
    (incomplete_tile / 'tile.pt').write_text('')

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  sample_ids:\n'
        '    - TENX95\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_create_items_module()
    module.main('hest1k', config_path=config_path, overwrite=True)

    items_path = data_dir / '03_output' / 'hest1k' / 'items' / 'all.json'
    assert items_path.exists()
    assert (data_dir / '03_output' / 'hest1k' / 'panels' / 'default.yaml').exists()

    items = json.loads(items_path.read_text())
    assert items == [
        {
            'id': 'TENX95_0',
            'sample_id': 'TENX95',
            'tile_id': 0,
            'tile_dir': str(complete_tile),
        }
    ]
