import importlib.util
import json
import io
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger

from xenium_hne_fusion.config import ArtifactsConfig, ItemsConfig, ItemsThresholdConfig
from xenium_hne_fusion.pipeline import filter_items


def _load_filter_items_module():
    path = Path('scripts/artifacts/filter_items.py').resolve()
    spec = importlib.util.spec_from_file_location('filter_items_script', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_items_uses_beat_default_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'beat'
    output_dir = data_dir / '03_output' / 'beat'
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
    artifacts_cfg = ArtifactsConfig(
        name='beat',
        items=ItemsConfig(name='default', filter=ItemsThresholdConfig(num_transcripts=200)),
    )
    module.main(artifacts_cfg=artifacts_cfg, overwrite=True)

    filtered = json.loads((output_dir / 'items' / 'default.json').read_text())
    assert [item['id'] for item in filtered] == ['S1_1', 'S1_2']


def test_filter_items_filters_hest1k_by_organ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    output_dir = data_dir / '03_output' / 'hest1k'
    processed_dir = data_dir / '02_processed' / 'hest1k'
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
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='lung', filter=ItemsThresholdConfig(organs=['Lung'], num_transcripts=100)),
    )
    module.main(artifacts_cfg=artifacts_cfg, overwrite=True)

    filtered = json.loads((output_dir / 'items' / 'lung.json').read_text())
    assert [item['id'] for item in filtered] == ['L1_0']


def test_filter_items_derives_stats_from_items_stem(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'beat'
    items_path = output_dir / 'items' / 'subset.json'
    output_path = output_dir / 'items' / 'default.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)

    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/S1/0'},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/S1/1'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [199, 200],
            'num_unique_transcripts': [None, None],
            'num_cells': [None, None],
            'num_unique_cells': [None, None],
        },
        index=pd.Index(['S1_0', 'S1_1'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'subset.parquet')

    output_path, n_kept = filter_items(
        items_path=items_path,
        output_path=output_path,
        items_cfg=ItemsConfig(name='default', filter=ItemsThresholdConfig(num_transcripts=200)),
        overwrite=True,
    )

    assert output_path == output_dir / 'items' / 'default.json'
    assert n_kept == 1
    filtered = json.loads(output_path.read_text())
    assert [item['id'] for item in filtered] == ['S1_1']


def test_filter_items_supports_sample_exclude_ids(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'beat'
    items_path = output_dir / 'items' / 'all.json'
    output_path = output_dir / 'items' / 'default.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)

    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/S1/0'},
                {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': '/tmp/S2/0'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [200, 200],
            'num_unique_transcripts': [None, None],
            'num_cells': [None, None],
            'num_unique_cells': [None, None],
        },
        index=pd.Index(['S1_0', 'S2_0'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'all.parquet')

    output_path, n_kept = filter_items(
        items_path=items_path,
        output_path=output_path,
        items_cfg=ItemsConfig(name='default', filter=ItemsThresholdConfig(exclude_ids=['S2'], num_transcripts=200)),
        overwrite=True,
    )

    assert output_path == output_dir / 'items' / 'default.json'
    assert n_kept == 1
    filtered = json.loads(output_path.read_text())
    assert [item['id'] for item in filtered] == ['S1_0']


def test_filter_items_drops_items_with_missing_threshold_stats(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'beat'
    items_path = output_dir / 'items' / 'all.json'
    output_path = output_dir / 'items' / 'default.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)

    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/S1/0'},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/S1/1'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [200, None],
            'num_unique_transcripts': [None, None],
            'num_cells': [None, None],
            'num_unique_cells': [None, None],
        },
        index=pd.Index(['S1_0', 'S1_1'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'all.parquet')

    output_path, n_kept = filter_items(
        items_path=items_path,
        output_path=output_path,
        items_cfg=ItemsConfig(name='default', filter=ItemsThresholdConfig(num_transcripts=200)),
        overwrite=True,
    )

    assert output_path == output_dir / 'items' / 'default.json'
    assert n_kept == 1
    filtered = json.loads(output_path.read_text())
    assert [item['id'] for item in filtered] == ['S1_0']


def test_filter_items_logs_stage_counts(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'beat'
    items_path = output_dir / 'items' / 'all.json'
    output_path = output_dir / 'items' / 'default.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / 'statistics').mkdir(parents=True, exist_ok=True)

    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/S1/0'},
                {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': '/tmp/S2/0'},
            ]
        )
    )
    pd.DataFrame(
        {
            'num_transcripts': [200, 199],
            'num_unique_transcripts': [None, None],
            'num_cells': [None, None],
            'num_unique_cells': [None, None],
        },
        index=pd.Index(['S1_0', 'S2_0'], name='id'),
    ).to_parquet(output_dir / 'statistics' / 'all.parquet')

    sink = io.StringIO()
    sink_id = logger.add(sink, level='INFO')
    try:
        output_path, n_kept = filter_items(
            items_path=items_path,
            output_path=output_path,
            items_cfg=ItemsConfig(name='default', filter=ItemsThresholdConfig(num_transcripts=200)),
            overwrite=True,
        )
    finally:
        logger.remove(sink_id)

    assert output_path == output_dir / 'items' / 'default.json'
    assert n_kept == 1
    log_output = sink.getvalue()
    assert 'Loaded 2 items from' in log_output
    assert 'Stats filter all.parquet: 2 -> 1 tiles (1 removed)' in log_output
