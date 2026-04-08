import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from xenium_hne_fusion.hvg import (
    build_hvg_anndata,
    build_tile_level_matrix,
    create_panel,
    get_common_genes,
    load_fit_items,
)
from xenium_hne_fusion.utils.getters import load_processing_config


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_transcripts_parquet(tile_dir: Path, genes: list[str], observed_genes: list[str]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            'feature_name': pd.Categorical(
                observed_genes,
                categories=genes,
                ordered=False,
            )
        }
    ).to_parquet(tile_dir / 'transcripts.parquet')


def test_build_tile_level_matrix_counts_tile_transcripts_and_preserves_gene_order(tmp_path: Path):
    tile_dir = tmp_path / 'S1' / '256_256' / '0'
    _write_transcripts_parquet(tile_dir, ['A', 'B', 'C'], ['A', 'C', 'C', 'A', 'A', 'B'])

    fit_items = pd.DataFrame(
        [{'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_dir)}]
    )
    matrix, obs = build_tile_level_matrix(fit_items, ['A', 'B', 'C'])

    assert obs.index.tolist() == ['S1_0']
    assert matrix.shape == (1, 3)
    assert matrix.toarray().tolist() == [[3.0, 1.0, 2.0]]


def test_load_fit_items_filters_to_fit_split(tmp_path: Path):
    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': '/tmp/a'},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': '/tmp/b'},
                {'id': 'S1_2', 'sample_id': 'S1', 'tile_id': 2, 'tile_dir': '/tmp/c'},
            ]
        )
    )
    pd.DataFrame(
        {'split': ['fit', 'val', 'test']},
        index=pd.Index(['S1_0', 'S1_1', 'S1_2'], name='id'),
    ).to_parquet(split_path)

    fit_items = load_fit_items(items_path, split_path)

    assert fit_items['id'].tolist() == ['S1_0']


def test_get_common_genes_uses_intersection_across_samples(tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S2' / '256_256' / '0'
    _write_transcripts_parquet(tile_a, ['A', 'B', 'C'], ['A'])
    _write_transcripts_parquet(tile_b, ['B', 'C', 'D'], ['B'])

    fit_items = pd.DataFrame(
        [
            {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_a)},
            {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': str(tile_b)},
        ]
    )

    assert get_common_genes(fit_items) == ['B', 'C']


def test_create_panel_writes_target_hvgs_and_source_remainder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S1' / '256_256' / '1'
    _write_transcripts_parquet(tile_a, ['A', 'B', 'C'], ['A', 'C', 'C', 'B'])
    _write_transcripts_parquet(tile_b, ['A', 'B', 'C'], ['B', 'B', 'B', 'A'])

    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    output_path = tmp_path / 'panels' / 'hvg-default.yaml'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_a)},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': str(tile_b)},
            ]
        )
    )
    pd.DataFrame(
        {'split': ['fit', 'fit']},
        index=pd.Index(['S1_0', 'S1_1'], name='id'),
    ).to_parquet(split_path)

    def fake_hvg(adata, n_top_genes, flavor, inplace):
        adata.var['highly_variable'] = [False, True, True]

    monkeypatch.setattr('scanpy.pp.highly_variable_genes', fake_hvg)

    create_panel(
        items_path=items_path,
        split_metadata_path=split_path,
        output_path=output_path,
        n_top_genes=2,
        overwrite=True,
    )

    panel = __import__('yaml').safe_load(output_path.read_text())
    assert panel['source_panel'] == ['A']
    assert panel['target_panel'] == ['B', 'C']


def test_create_panel_script_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    output_dir = data_dir / '03_output' / 'hest1k'
    tile_dir = data_dir / '02_processed' / 'hest1k' / 'TENX95' / '256_256' / '0'
    config_path = tmp_path / 'hest1k.yaml'
    tile_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    (output_dir / 'splits' / 'default').mkdir(parents=True, exist_ok=True)

    _write_transcripts_parquet(tile_dir, ['A', 'B'], ['A', 'B', 'B', 'B'])
    items = json.dumps([{'id': 'TENX95_0', 'sample_id': 'TENX95', 'tile_id': 0, 'tile_dir': str(tile_dir)}])
    (output_dir / 'items' / 'all.json').write_text(items)
    (output_dir / 'items' / 'default.json').write_text(items)
    pd.DataFrame(
        {'split': ['fit']},
        index=pd.Index(['TENX95_0'], name='id'),
    ).to_parquet(output_dir / 'splits' / 'default' / 'outer=0-seed=0.parquet')

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids: null\n'
        '  exclude_ids: null\n'
        'items:\n'
        '  name: default\n'
        'split:\n'
        '  name: default\n'
        '  random_state: 0\n'
        'panel:\n'
        '  name: hvg-default-default-outer=0-seed=0\n'
        '  n_top_genes: 1\n'
        '  flavor: seurat_v3\n'
    )

    def fake_hvg(adata, n_top_genes, flavor, inplace):
        adata.var['highly_variable'] = [False, True]

    monkeypatch.setattr('scanpy.pp.highly_variable_genes', fake_hvg)
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_panel.py', 'create_panel_script')
    processing_cfg = load_processing_config(config_path)
    module.main(processing_cfg=processing_cfg, overwrite=True)

    panel_path = output_dir / 'panels' / 'hvg-default-default-outer=0-seed=0.yaml'
    assert panel_path.exists()
    panel = __import__('yaml').safe_load(panel_path.read_text())
    assert panel['source_panel'] == ['A']
    assert panel['target_panel'] == ['B']


def test_create_panel_script_accepts_predefined_panel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    output_dir = data_dir / '03_output' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'

    (output_dir / 'panels').mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / 'panels' / 'default.yaml'
    panel_path.write_text(__import__('yaml').safe_dump({'source_panel': ['A'], 'target_panel': ['B']}, sort_keys=False))
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'items:\n'
        '  name: default\n'
        'split:\n'
        '  name: default\n'
        'panel:\n'
        '  name: default\n'
        '  n_top_genes: null\n'
        '  flavor: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_panel.py', 'create_panel_predefined_script')
    processing_cfg = load_processing_config(config_path)
    module.main(processing_cfg=processing_cfg, overwrite=False)

    assert panel_path.exists()


def test_create_panel_script_rejects_missing_predefined_panel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'items:\n'
        '  name: default\n'
        'split:\n'
        '  name: default\n'
        'panel:\n'
        '  name: missing\n'
        '  n_top_genes: null\n'
        '  flavor: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_panel.py', 'create_panel_missing_predefined_script')
    processing_cfg = load_processing_config(config_path)
    with pytest.raises(AssertionError, match='Panel not found'):
        module.main(processing_cfg=processing_cfg, overwrite=False)


def test_create_panel_script_rejects_mixed_panel_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'items:\n'
        '  name: default\n'
        'split:\n'
        '  name: default\n'
        'panel:\n'
        '  name: default\n'
        '  n_top_genes: 128\n'
        '  flavor: null\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    module = _load_script('scripts/data/create_panel.py', 'create_panel_mixed_mode_script')
    processing_cfg = load_processing_config(config_path)
    with pytest.raises(AssertionError, match='panel config must set both n_top_genes and flavor'):
        module.main(processing_cfg=processing_cfg, overwrite=False)


def test_build_hvg_anndata_uses_one_row_per_tile(tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S1' / '256_256' / '1'
    _write_transcripts_parquet(tile_a, ['A', 'B'], ['A', 'A', 'B'])
    _write_transcripts_parquet(tile_b, ['A', 'B'], ['B', 'B', 'B', 'B', 'A'])

    fit_items = pd.DataFrame(
        [
            {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_a)},
            {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': str(tile_b)},
        ]
    )

    adata = build_hvg_anndata(fit_items)

    assert adata.n_obs == 2
    assert adata.obs_names.tolist() == ['S1_0', 'S1_1']
    assert adata.var_names.tolist() == ['A', 'B']
