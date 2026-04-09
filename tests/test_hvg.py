import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from xenium_hne_fusion.config import ArtifactsConfig, ItemsConfig, PanelConfig, SplitConfig
from xenium_hne_fusion.hvg import (
    build_hvg_anndata_from_split,
    create_panel,
    get_common_genes,
)
from xenium_hne_fusion.datasets.tiles import TileDataset


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_expr_parquet(tile_dir: Path, genes: list[str], rows: list[list[int]]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=genes).to_parquet(tile_dir / 'expr-kernel_size=16.parquet', index=False)


def _write_feature_universe(tile_dir: Path, genes: list[str]) -> None:
    sample_dir = tile_dir.parent.parent
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / 'feature_universe.txt').write_text('\n'.join(genes) + '\n')


def test_build_hvg_anndata_from_split_counts_tile_expression_and_preserves_gene_order(tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S1' / '256_256' / '1'
    _write_expr_parquet(tile_a, ['A', 'B', 'C'], [[1, 0, 2], [2, 1, 0]])
    _write_expr_parquet(tile_b, ['A', 'B', 'C'], [[0, 1, 1], [1, 0, 1]])
    _write_feature_universe(tile_a, ['A', 'B', 'C'])

    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_a)},
                {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': str(tile_b)},
            ]
        )
    )
    pd.DataFrame(
        {'split': ['fit', 'fit'], 'sample_id': ['S1', 'S1']},
        index=pd.Index(['S1_0', 'S1_1'], name='id'),
    ).to_parquet(split_path)

    adata = build_hvg_anndata_from_split(
        items_path=items_path,
        split_metadata_path=split_path,
        genes=['A', 'B', 'C'],
    )

    assert adata.obs.index.tolist() == ['S1_0', 'S1_1']
    assert adata.X.shape == (2, 3)
    assert adata.var_names.tolist() == ['A', 'B', 'C']
    assert adata.X.toarray().tolist() == [[3.0, 1.0, 2.0], [1.0, 1.0, 2.0]]


def test_tile_dataset_filters_to_fit_split(tmp_path: Path):
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

    ds = TileDataset(
        target='expression',
        source_panel=[],
        target_panel=[],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        metadata_path=split_path,
        split='fit',
        id_key='id',
    )
    ds.setup()

    assert [item['id'] for item in ds.items] == ['S1_0']


def test_get_common_genes_uses_intersection_across_samples(tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S2' / '256_256' / '0'
    _write_expr_parquet(tile_a, ['A', 'B', 'C'], [[1, 0, 0]])
    _write_expr_parquet(tile_b, ['B', 'C', 'D'], [[1, 0, 0]])
    _write_feature_universe(tile_a, ['A', 'B', 'C'])
    _write_feature_universe(tile_b, ['B', 'C', 'D'])

    fit_items = pd.DataFrame(
        [
            {'id': 'S1_0', 'sample_id': 'S1', 'split': 'fit'},
            {'id': 'S2_0', 'sample_id': 'S2', 'split': 'fit'},
        ]
    )

    assert get_common_genes(fit_items, processed_dir=tmp_path) == ['B', 'C']


def test_create_panel_writes_target_hvgs_and_source_remainder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S1' / '256_256' / '1'
    _write_expr_parquet(tile_a, ['A', 'B', 'C'], [[1, 0, 2], [0, 1, 1]])
    _write_expr_parquet(tile_b, ['A', 'B', 'C'], [[2, 1, 0], [1, 2, 0]])
    _write_feature_universe(tile_a, ['A', 'B', 'C'])

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
        {'split': ['fit', 'fit'], 'sample_id': ['S1', 'S1']},
        index=pd.Index(['S1_0', 'S1_1'], name='id'),
    ).to_parquet(split_path)

    def fake_hvg(adata, n_top_genes, flavor, inplace):
        adata.var['highly_variable'] = [False, True, True]

    monkeypatch.setattr('scanpy.pp.highly_variable_genes', fake_hvg)

    create_panel(
        items_path=items_path,
        split_metadata_path=split_path,
        processed_dir=tmp_path,
        output_path=output_path,
        n_top_genes=2,
        overwrite=True,
    )

    panel = __import__('yaml').safe_load(output_path.read_text())
    assert panel['source_panel'] == ['A']
    assert panel['target_panel'] == ['B', 'C']


def test_create_panel_uses_batched_tile_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tile_a = tmp_path / 'S1' / '256_256' / '0'
    tile_b = tmp_path / 'S2' / '256_256' / '0'
    _write_expr_parquet(tile_a, ['A', 'B'], [[1, 0], [0, 1]])
    _write_expr_parquet(tile_b, ['A', 'B'], [[0, 1], [1, 0]])
    _write_feature_universe(tile_a, ['A', 'B'])
    _write_feature_universe(tile_b, ['A', 'B'])

    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    output_path = tmp_path / 'panels' / 'hvg-default.yaml'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_a)},
                {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': str(tile_b)},
            ]
        )
    )
    pd.DataFrame(
        {
            'split': ['fit', 'fit'],
            'sample_id': ['S1', 'S2'],
        },
        index=pd.Index(['S1_0', 'S2_0'], name='id'),
    ).to_parquet(split_path)

    real_loader = torch.utils.data.DataLoader
    calls = {}

    def fake_loader(dataset, *, batch_size, num_workers, shuffle):
        calls['batch_size'] = batch_size
        calls['num_workers'] = num_workers
        calls['shuffle'] = shuffle
        return real_loader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def fake_hvg(adata, n_top_genes, flavor, inplace):
        adata.var['highly_variable'] = [False, True]

    monkeypatch.setattr('xenium_hne_fusion.hvg.DataLoader', fake_loader)
    monkeypatch.setattr('scanpy.pp.highly_variable_genes', fake_hvg)

    create_panel(
        items_path=items_path,
        split_metadata_path=split_path,
        processed_dir=tmp_path,
        output_path=output_path,
        n_top_genes=1,
        overwrite=True,
    )

    assert calls == {'batch_size': 256, 'num_workers': 10, 'shuffle': False}


def test_create_panel_rejects_when_common_genes_are_fewer_than_requested(tmp_path: Path):
    tile_dir = tmp_path / 'S1' / '256_256' / '0'
    _write_expr_parquet(tile_dir, ['A'], [[1], [0]])
    _write_feature_universe(tile_dir, ['A'])

    items_path = tmp_path / 'items.json'
    split_path = tmp_path / 'default.parquet'
    output_path = tmp_path / 'panels' / 'hvg-default.yaml'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(tile_dir)},
            ]
        )
    )
    pd.DataFrame(
        {
            'split': ['fit'],
            'sample_id': ['S1'],
        },
        index=pd.Index(['S1_0'], name='id'),
    ).to_parquet(split_path)

    with pytest.raises(AssertionError, match='exceeds common genes'):
        create_panel(
            items_path=items_path,
            split_metadata_path=split_path,
            processed_dir=tmp_path,
            output_path=output_path,
            n_top_genes=2,
            overwrite=True,
        )


def test_create_panel_script_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    output_dir = data_dir / '03_output' / 'hest1k'
    tile_dir = data_dir / '02_processed' / 'hest1k' / 'TENX95' / '256_256' / '0'
    config_path = tmp_path / 'hest1k.yaml'
    tile_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'items').mkdir(parents=True, exist_ok=True)
    (output_dir / 'splits' / 'default').mkdir(parents=True, exist_ok=True)

    _write_expr_parquet(tile_dir, ['A', 'B'], [[1, 0], [0, 1], [0, 1], [0, 1]])
    _write_feature_universe(tile_dir, ['A', 'B'])
    items = json.dumps([{'id': 'TENX95_0', 'sample_id': 'TENX95', 'tile_id': 0, 'tile_dir': str(tile_dir)}])
    (output_dir / 'items' / 'all.json').write_text(items)
    (output_dir / 'items' / 'default.json').write_text(items)
    pd.DataFrame(
        {
            'split': ['fit'],
            'sample_id': ['TENX95'],
        },
        index=pd.Index(['TENX95_0'], name='id'),
    ).to_parquet(output_dir / 'splits' / 'default' / 'outer=0-seed=0.parquet')

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'img_size: 224\n'
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

    module = _load_script('scripts/artifacts/create_panel.py', 'create_panel_script')
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default', random_state=0),
        panel=PanelConfig(
            name='hvg-default-default-outer=0-seed=0',
            metadata_path=Path('default/outer=0-seed=0.parquet'),
            n_top_genes=1,
            flavor='seurat_v3',
        ),
    )
    module.main(artifacts_cfg=artifacts_cfg, overwrite=True)

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
        'img_size: 224\n'
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

    module = _load_script('scripts/artifacts/create_panel.py', 'create_panel_predefined_script')
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default'),
        panel=PanelConfig(name='default'),
    )
    module.main(artifacts_cfg=artifacts_cfg, overwrite=False)

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
        'img_size: 224\n'
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

    module = _load_script('scripts/artifacts/create_panel.py', 'create_panel_missing_predefined_script')
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default'),
        panel=PanelConfig(name='missing'),
    )
    with pytest.raises(AssertionError, match='Panel not found'):
        module.main(artifacts_cfg=artifacts_cfg, overwrite=False)


def test_create_panel_script_rejects_mixed_panel_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'

    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'img_size: 224\n'
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

    module = _load_script('scripts/artifacts/create_panel.py', 'create_panel_mixed_mode_script')
    artifacts_cfg = ArtifactsConfig(
        name='hest1k',
        items=ItemsConfig(name='default'),
        split=SplitConfig(name='default'),
        panel=PanelConfig(name='default', n_top_genes=128),
    )
    with pytest.raises(AssertionError, match='panel.flavor is required'):
        module.main(artifacts_cfg=artifacts_cfg, overwrite=False)
