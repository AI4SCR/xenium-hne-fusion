from pathlib import Path

import pytest

from xenium_hne_fusion.utils.getters import (
    build_pipeline_config,
    load_dataset_config,
    load_pipeline_config,
    resolve_samples,
    select_sample_ids,
)


def test_load_dataset_config_requires_name(tmp_path: Path):
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids:\n'
        '    - TENX95\n'
    )

    with pytest.raises(KeyError):
        load_dataset_config(config_path)


def test_load_pipeline_config_resolves_name_scoped_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / 'hest1k'
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids:\n'
        '    - TENX95\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(raw_dir))

    cfg = load_pipeline_config('hest1k', config_path)

    assert cfg.name == 'hest1k'
    assert cfg.raw_dir == raw_dir.resolve()
    assert cfg.structured_dir == (data_dir / '01_structured' / 'hest1k').resolve()
    assert cfg.processed_dir == (data_dir / '02_processed' / 'hest1k').resolve()
    assert cfg.output_dir == (data_dir / '03_output' / 'hest1k').resolve()


def test_load_pipeline_config_requires_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_path = tmp_path / 'beat.yaml'
    config_path.write_text(
        'name: beat\n'
        'tile_px: 512\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids: null\n'
        '  exclude_ids: null\n'
    )

    monkeypatch.delenv('DATA_DIR', raising=False)
    monkeypatch.delenv('BEAT_RAW_DIR', raising=False)
    with pytest.raises(AssertionError, match='DATA_DIR'):
        load_pipeline_config('beat', config_path)

    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    with pytest.raises(AssertionError, match='BEAT_RAW_DIR'):
        load_pipeline_config('beat', config_path)


@pytest.mark.parametrize(
    ('dataset', 'env_var'),
    [('beat', 'BEAT_RAW_DIR'), ('hest1k', 'HEST1K_RAW_DIR')],
)
def test_build_pipeline_config_uses_exact_dataset_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: str,
    env_var: str,
):
    data_dir = tmp_path / 'data'
    raw_dir = tmp_path / 'raw' / dataset
    config_path = tmp_path / f'{dataset}.yaml'
    config_path.write_text(
        f'name: {dataset}\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
    )

    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv(env_var, str(raw_dir))

    cfg = build_pipeline_config(load_dataset_config(config_path))

    assert cfg.dataset == dataset
    assert cfg.raw_dir == raw_dir.resolve()


@pytest.mark.parametrize('dataset', ['foo', 'beat-256', 'hest1k-anything'])
def test_build_pipeline_config_rejects_non_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: str,
):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        f'name: {dataset}\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
    )

    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('BEAT_RAW_DIR', str(tmp_path / 'raw' / 'beat'))
    monkeypatch.setenv('HEST1K_RAW_DIR', str(tmp_path / 'raw' / 'hest1k'))

    with pytest.raises(AssertionError, match='Unknown dataset'):
        build_pipeline_config(load_dataset_config(config_path))


def test_resolve_samples_supports_hest_metadata_columns(tmp_path: Path):
    config_path = tmp_path / 'hest1k.yaml'
    metadata_path = tmp_path / 'HEST_v1_3_0.csv'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  species: Homo sapiens\n'
        '  disease_type: Cancer\n'
        '  include_ids: null\n'
        '  exclude_ids: null\n'
    )
    metadata_path.write_text(
        'id,st_technology,species,organ,disease_state\n'
        'TENX95,Xenium,Homo sapiens,Breast,Cancer\n'
        'TENX96,Visium,Homo sapiens,Breast,Cancer\n'
        'TENX97,Xenium,Mus musculus,Breast,Cancer\n'
    )

    cfg = load_dataset_config(config_path)

    assert resolve_samples(cfg, metadata_path) == ['TENX95']


def test_load_processing_config_loads_split_name_and_panel(tmp_path: Path):
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'split:\n'
        '  name: default\n'
        'panel:\n'
        '  name: default\n'
        '  n_top_genes: 128\n'
        '  flavor: seurat_v3\n'
    )

    cfg = load_dataset_config(config_path)

    assert cfg.split.name == 'default'
    assert cfg.panel is not None
    assert cfg.panel.name == 'default'
    assert cfg.panel.n_top_genes == 128
    assert cfg.panel.flavor == 'seurat_v3'


def test_repo_hest_config_pins_three_include_ids():
    cfg = load_dataset_config(Path('configs/data/local/hest1k.yaml'))

    assert cfg.filter.include_ids == ['NCBI783', 'NCBI856', 'TENX116']
    assert cfg.filter.exclude_ids is None
    assert cfg.items.filter.include_ids is None
    assert cfg.items.filter.exclude_ids is None


def test_repo_remote_hest_config_processes_all_samples():
    cfg = load_dataset_config(Path('configs/data/remote/hest1k.yaml'))

    assert cfg.filter.include_ids is None
    assert cfg.filter.exclude_ids is None
    assert cfg.items.filter.include_ids is None
    assert cfg.items.filter.exclude_ids is None


@pytest.mark.parametrize(
    ('path', 'items_name', 'split_name', 'panel_name', 'n_top_genes'),
    [
        ('configs/data/local/hest1k-breast.yaml', 'breast', 'breast', 'hvg-breast-breast-outer=0-seed=0', 16),
        ('configs/data/local/hest1k-lung.yaml', 'lung', 'lung', 'hvg-lung-lung-outer=0-seed=0', 8),
        ('configs/data/local/hest1k-pancreas.yaml', 'pancreas', 'pancreas', 'hvg-pancreas-pancreas-outer=0-seed=0', 128),
        ('configs/data/remote/hest1k-breast.yaml', 'breast', 'breast', 'hvg-breast-breast-outer=0-seed=0', 16),
        ('configs/data/remote/hest1k-lung.yaml', 'lung', 'lung', 'hvg-lung-lung-outer=0-seed=0', 8),
        ('configs/data/remote/hest1k-pancreas.yaml', 'pancreas', 'pancreas', 'hvg-pancreas-pancreas-outer=0-seed=0', 128),
    ],
)
def test_repo_hest_organ_configs_are_fully_specified(
    path: str,
    items_name: str,
    split_name: str,
    panel_name: str,
    n_top_genes: int,
):
    cfg = load_dataset_config(Path(path))

    assert cfg.name == 'hest1k'
    assert cfg.items.name == items_name
    assert cfg.split.name == split_name
    assert cfg.panel is not None
    assert cfg.panel.name == panel_name
    assert cfg.panel.n_top_genes == n_top_genes
    assert cfg.panel.flavor == 'seurat_v3'


def test_items_filter_sample_ids_override_legacy_top_level_filter(tmp_path: Path):
    config_path = tmp_path / 'hest1k.yaml'
    config_path.write_text(
        'name: hest1k\n'
        'tile_px: 256\n'
        'stride_px: 256\n'
        'tile_mpp: 0.5\n'
        'filter:\n'
        '  include_ids:\n'
        '    - LEGACY\n'
        '  exclude_ids: null\n'
        'items:\n'
        '  name: default\n'
        '  filter:\n'
        '    include_ids: null\n'
        '    exclude_ids:\n'
        '      - CANONICAL\n'
    )

    cfg = load_dataset_config(config_path)

    assert cfg.filter.include_ids == ['LEGACY']
    assert cfg.filter.exclude_ids is None
    assert cfg.items.filter.include_ids is None
    assert cfg.items.filter.exclude_ids == ['CANONICAL']


def test_select_sample_ids_supports_include_and_exclude():
    cfg = load_dataset_config(Path('configs/data/remote/hest1k.yaml'))
    cfg.filter.include_ids = ['S2', 'S1']
    assert select_sample_ids(['S1', 'S2', 'S3'], cfg.filter) == ['S1', 'S2']

    cfg.filter.include_ids = None
    cfg.filter.exclude_ids = ['S2']
    assert select_sample_ids(['S1', 'S2', 'S3'], cfg.filter) == ['S1', 'S3']


def test_select_sample_ids_rejects_mutually_exclusive_filters():
    cfg = load_dataset_config(Path('configs/data/remote/hest1k.yaml'))
    cfg.filter.include_ids = ['S1']
    cfg.filter.exclude_ids = ['S2']

    with pytest.raises(AssertionError, match='mutually exclusive'):
        select_sample_ids(['S1', 'S2'], cfg.filter)
