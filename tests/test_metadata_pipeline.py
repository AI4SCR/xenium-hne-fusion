import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.metadata import link_structured_metadata
from xenium_hne_fusion.utils.getters import load_processing_config


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_process_metadata_writes_cleaned_sample_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    structured_dir = data_dir / '01_structured' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'
    raw_metadata_path = raw_dir / 'HEST_v1_3_0.csv'

    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {'id': 'TENX95', 'st_technology': 'Xenium', 'species': 'Homo sapiens', 'organ': 'Breast', 'disease_state': 'Cancer'},
            {'id': 'TENX96', 'st_technology': 'Xenium', 'species': 'Homo sapiens', 'organ': 'Lung', 'disease_state': 'Healthy'},
            {'id': 'VIS1', 'st_technology': 'Visium', 'species': 'Homo sapiens', 'organ': 'Breast', 'disease_state': 'Cancer'},
        ]
    ).to_csv(raw_metadata_path, index=False)
    link_structured_metadata(raw_metadata_path, structured_dir)

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  species: Homo sapiens\n'
        '  include_ids:\n'
        '    - TENX95\n'
        '    - TENX96\n'
        '  exclude_ids: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/process_metadata.py', 'process_metadata_script')
    module.main(load_processing_config(config_path))

    output_path = data_dir / '02_processed' / 'hest1k' / 'metadata.parquet'
    assert output_path.exists()

    metadata = pd.read_parquet(output_path)
    assert metadata['sample_id'].tolist() == ['TENX95', 'TENX96']
    assert 'id' not in metadata.columns


def test_process_metadata_writes_beat_metadata_to_processed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'beat'
    structured_dir = data_dir / '01_structured' / 'beat'
    config_path = tmp_path / 'beat.yaml'
    raw_metadata_path = raw_dir / 'metadata.parquet'

    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {'sample_id': 'S1', 'patient': 'P1'},
            {'sample_id': 'S2', 'patient': 'P2'},
        ]
    ).set_index('sample_id').to_parquet(raw_metadata_path)
    link_structured_metadata(raw_metadata_path, structured_dir)

    config_path.write_text(
        'name: beat\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids:\n'
        '    - S2\n'
        '  exclude_ids: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('BEAT_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/process_metadata.py', 'process_metadata_beat_script')
    module.main(load_processing_config(config_path))

    output_path = data_dir / '02_processed' / 'beat' / 'metadata.parquet'
    assert output_path.exists()

    metadata = pd.read_parquet(output_path)
    assert metadata['sample_id'].tolist() == ['S2']


def test_create_splits_writes_tile_level_metadata_with_sample_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
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
        '  include_ids: null\n'
        '  exclude_ids: null\n'
    )
    sample_rows = []
    items = []
    for sample_idx in range(9):
        sample_id = f'S{sample_idx}'
        sample_rows.append({'sample_id': sample_id, 'patient': f'P{sample_idx}', 'cohort': sample_idx % 2})
        for tile_id in range(2):
            items.append(
                {
                    'id': f'{sample_id}_{tile_id}',
                    'sample_id': sample_id,
                    'tile_id': tile_id,
                    'tile_dir': f'/tmp/{sample_id}/{tile_id}',
                }
            )

    processed_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_parquet(processed_dir / 'metadata.parquet', index=False)
    (output_dir / 'items' / 'all.json').write_text(json.dumps(items))

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_splits.py', 'create_splits_script')
    processing_cfg = load_processing_config(config_path)
    processing_cfg.split.split_name = 'default'
    processing_cfg.split.test_size = 0.2
    processing_cfg.split.val_size = None
    processing_cfg.split.stratify = False
    processing_cfg.split.group_column_name = 'sample_id'
    processing_cfg.split.random_state = 0
    module.main('hest1k', config_path=None, overwrite=True, processing_cfg=processing_cfg)

    split_dir = output_dir / 'splits' / 'default'
    assert split_dir.exists()

    split_metadata = pd.read_parquet(split_dir / 'outer=0-seed=0.parquet')
    assert set(split_metadata.index) == {item['id'] for item in items}
    assert {'sample_id', 'tile_id', 'tile_dir', 'patient', 'cohort', 'split'} <= set(split_metadata.columns)

    for _, group in split_metadata.groupby('sample_id'):
        assert group['split'].nunique() == 1


def test_beat_default_split_groups_by_sample_id():
    cfg = load_processing_config(Path('configs/data/local/beat.yaml'))
    assert cfg.split.group_column_name == 'sample_id'


def test_create_hest1k_organ_splits_can_mix_tiles_within_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
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
        '  include_ids: null\n'
        '  exclude_ids: null\n'
    )
    sample_rows = []
    items = []
    for sample_idx in range(2):
        sample_id = f'S{sample_idx}'
        sample_rows.append({'sample_id': sample_id, 'organ': 'Lung'})
        for tile_id in range(20):
            items.append(
                {
                    'id': f'{sample_id}_{tile_id}',
                    'sample_id': sample_id,
                    'tile_id': tile_id,
                    'tile_dir': f'/tmp/{sample_id}/{tile_id}',
                }
            )

    processed_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_parquet(processed_dir / 'metadata.parquet', index=False)
    (output_dir / 'items' / 'lung.json').write_text(json.dumps(items))

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_splits.py', 'create_splits_lung_script')
    processing_cfg = load_processing_config(config_path)
    processing_cfg.split.split_name = 'lung'
    processing_cfg.split.test_size = 0.2
    processing_cfg.split.val_size = None
    processing_cfg.split.stratify = False
    processing_cfg.split.group_column_name = None
    processing_cfg.split.random_state = 0
    module.main(
        'hest1k',
        config_path=None,
        items_path=output_dir / 'items' / 'lung.json',
        overwrite=True,
        processing_cfg=processing_cfg,
    )

    split_path = output_dir / 'splits' / 'lung' / 'outer=0-seed=0.parquet'
    assert split_path.exists()

    split_metadata = pd.read_parquet(split_path)
    assert set(split_metadata['sample_id']) == {'S0', 'S1'}
    assert split_metadata.groupby('sample_id')['split'].nunique().max() > 1
