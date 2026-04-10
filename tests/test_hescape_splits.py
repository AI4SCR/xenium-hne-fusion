import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_hescape_split_script_smoke(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    output_dir = data_dir / '03_output' / 'hest1k'
    processed_dir = data_dir / '02_processed' / 'hest1k'

    output_dir.joinpath('items').mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / 'items' / 'all.json').write_text(
        json.dumps(
            [
                {'id': 'TENX95_0', 'sample_id': 'TENX95', 'tile_id': 0, 'tile_dir': '/tmp/a'},
                {'id': 'NCBI785_0', 'sample_id': 'NCBI785', 'tile_id': 0, 'tile_dir': '/tmp/b'},
                {'id': 'NCBI783_0', 'sample_id': 'NCBI783', 'tile_id': 0, 'tile_dir': '/tmp/c'},
            ]
        )
    )
    pd.DataFrame(
        [
            {'sample_id': 'TENX95', 'organ': 'Breast'},
            {'sample_id': 'NCBI785', 'organ': 'Breast'},
            {'sample_id': 'NCBI783', 'organ': 'Breast'},
        ]
    ).to_parquet(processed_dir / 'metadata.parquet', index=False)

    monkeypatch.setenv('DATA_DIR', str(data_dir))

    module = _load_script('scripts/artifacts/create_hescape_splits.py', 'create_hescape_splits_script')
    monkeypatch.setattr(
        module,
        'PANEL_TO_SPLITS',
        {
            'hescape-breast': {
                'fit': {
                    4: 'TENX95',
                },
                'val': {
                    2: 'NCBI785',
                },
                'test': {
                    0: 'NCBI783',
                },
            }
        },
    )
    output_paths = module.create_hescape_splits(overwrite=True)
    split_path = output_dir / 'splits' / 'hescape' / 'hescape-breast.parquet'

    assert split_path.exists()
    assert split_path in output_paths
    split_metadata = pd.read_parquet(split_path)
    assert split_metadata.loc['TENX95_0', 'split'] == 'fit'
    assert split_metadata.loc['NCBI785_0', 'split'] == 'val'
    assert split_metadata.loc['NCBI783_0', 'split'] == 'test'
