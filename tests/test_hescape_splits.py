import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.metadata import load_named_split_ids, save_named_split_metadata


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_named_split_ids_rejects_overlap(tmp_path: Path):
    split_root = tmp_path / 'splits'
    split_root.mkdir()
    for split, ids in {
        'train': ['S1'],
        'val': ['S2'],
        'test': ['S1'],
    }.items():
        pd.DataFrame({'id': ids}).to_csv(split_root / f'{split}.csv', index=False)

    with pytest.raises(AssertionError, match='Overlap'):
        load_named_split_ids(split_root)


def test_save_named_split_metadata_assigns_tile_splits_from_sample_ids(tmp_path: Path):
    joined = pd.DataFrame(
        [
            {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/a'},
            {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/b'},
            {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': '/tmp/c'},
            {'id': 'S3_0', 'sample_id': 'S3', 'tile_id': 0, 'tile_dir': '/tmp/d'},
        ]
    ).set_index('id', drop=True)

    split_dir = tmp_path / 'out'
    save_named_split_metadata(
        joined,
        split_dir,
        {
            'train': ['S1'],
            'val': ['S2'],
            'test': ['S3'],
        },
        overwrite=False,
    )

    split_metadata = pd.read_parquet(split_dir / 'outer=0-inner=0-seed=0.parquet')
    assert split_metadata.loc['S1_0', 'split'] == 'fit'
    assert split_metadata.loc['S1_1', 'split'] == 'fit'
    assert split_metadata.loc['S2_0', 'split'] == 'val'
    assert split_metadata.loc['S3_0', 'split'] == 'test'


def test_cache_hescape_split_script_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    repo_root = tmp_path / 'repo'
    output_dir = data_dir / '03_output' / 'hest1k'
    processed_dir = data_dir / '02_processed' / 'hest1k'
    source_dir = repo_root / 'splits' / 'hest1k' / 'hescape' / 'human-breast-panel'
    config_path = tmp_path / 'hest1k.yaml'

    source_dir.mkdir(parents=True)
    output_dir.joinpath('items').mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'id': ['TENX95']}).to_csv(source_dir / 'train.csv', index=False)
    pd.DataFrame({'id': ['NCBI785']}).to_csv(source_dir / 'val.csv', index=False)
    pd.DataFrame({'id': ['NCBI783']}).to_csv(source_dir / 'test.csv', index=False)

    (output_dir / 'items' / 'breast.json').write_text(
        json.dumps(
            [
                {'id': 'NCBI783_0', 'sample_id': 'NCBI783', 'tile_id': 0, 'tile_dir': '/tmp/a'},
                {'id': 'NCBI783_1', 'sample_id': 'NCBI783', 'tile_id': 1, 'tile_dir': '/tmp/b'},
            ]
        )
    )
    pd.DataFrame(
        [
            {'sample_id': 'NCBI783', 'organ': 'Breast'},
        ]
    ).to_parquet(processed_dir / 'metadata.parquet', index=False)
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'img_size: 224\n'
        'filter:\n'
        '  include_ids: null\n'
        '  exclude_ids: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))
    monkeypatch.setenv('XHF_REPO_ROOT', str(repo_root))

    module = _load_script('scripts/data/cache_hescape_splits.py', 'cache_hescape_splits_script')
    split_dir = module.main('breast', config_path=config_path, overwrite=True)

    split_path = split_dir / 'outer=0-inner=0-seed=0.parquet'
    assert split_path.exists()
    split_metadata = pd.read_parquet(split_path)
    assert split_metadata['split'].tolist() == ['test', 'test']
