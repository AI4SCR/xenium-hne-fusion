from pathlib import Path

import pytest

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.utils import resolve_training_paths
from xenium_hne_fusion.utils.getters import get_panels_dir


def test_resolve_training_paths_requires_explicit_training_data_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('XHF_REPO_ROOT', str(tmp_path))

    cfg = Config()
    cfg.data.name = 'hest1k'

    with pytest.raises(AssertionError, match='cfg.data.items_path'):
        resolve_training_paths(cfg)


def test_resolve_training_paths_still_defaults_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'metadata.parquet'
    monkeypatch.setenv('DATA_DIR', str(data_dir))

    cfg = Config()
    cfg.data.name = 'hest1k'
    cfg.data.items_path = Path('all.json')
    cfg.data.metadata_path = metadata_path
    cfg.data.panel_path = Path('hvg-default-default-outer=0-seed=0.yaml')

    cfg, output_dir = resolve_training_paths(cfg)

    assert output_dir == (data_dir / '03_output' / 'hest1k').resolve()
    assert cfg.data.items_path == output_dir / 'items/all.json'
    assert cfg.data.cache_dir == output_dir / 'cache'
    assert cfg.data.metadata_path == metadata_path
    assert cfg.data.panel_path == output_dir / 'panels/hvg-default-default-outer=0-seed=0.yaml'


def test_resolve_training_paths_resolves_relative_paths_under_dataset_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    monkeypatch.setenv('DATA_DIR', str(data_dir))

    cfg = Config()
    cfg.data.name = 'beat'
    cfg.data.items_path = Path('train.json')
    cfg.data.metadata_path = Path('default/outer=0-seed=0.parquet')
    cfg.data.panel_path = Path('default.yaml')
    cfg.data.cache_dir = Path('run-a')

    cfg, output_dir = resolve_training_paths(cfg)

    assert cfg.data.items_path == output_dir / 'items/train.json'
    assert cfg.data.metadata_path == output_dir / 'splits' / 'default/outer=0-seed=0.parquet'
    assert cfg.data.panel_path == get_panels_dir('beat') / 'default.yaml'
    assert cfg.data.cache_dir == output_dir / 'cache/run-a'


def test_resolve_training_paths_keeps_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'custom' / 'split.parquet'
    items_path = tmp_path / 'custom' / 'items.json'
    panel_path = tmp_path / 'custom' / 'panel.yaml'
    cache_dir = tmp_path / 'custom' / 'cache'
    monkeypatch.setenv('DATA_DIR', str(data_dir))

    cfg = Config()
    cfg.data.name = 'beat'
    cfg.data.metadata_path = metadata_path
    cfg.data.items_path = items_path
    cfg.data.panel_path = panel_path
    cfg.data.cache_dir = cache_dir

    cfg, _ = resolve_training_paths(cfg)

    assert cfg.data.items_path == items_path
    assert cfg.data.metadata_path == metadata_path
    assert cfg.data.panel_path == panel_path
    assert cfg.data.cache_dir == cache_dir


def test_resolve_training_paths_requires_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    cfg = Config()
    cfg.data.metadata_path = tmp_path / 'metadata.parquet'

    with pytest.raises(AssertionError, match='cfg.data.name'):
        resolve_training_paths(cfg)


def test_train_configs_load_explicit_data_head_wandb_and_trainer_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))

    for path in Path('configs/train').rglob('*.yaml'):
        cfg = Config.from_yaml(path)
        assert cfg.task.target is not None
        assert cfg.data.name is not None
        assert isinstance(cfg.data.items_path, Path)
        assert isinstance(cfg.data.metadata_path, Path)
        assert isinstance(cfg.data.panel_path, Path)
        assert cfg.head.num_hidden_layers == 0
        assert cfg.lit.target_key is not None
        assert cfg.wandb.project is not None
        assert cfg.trainer.max_time == '00:01:45:00'
        assert cfg.trainer.max_epochs == 35
        assert cfg.trainer.gradient_clip_val == 1.0


def test_train_configs_use_expected_panel_defaults():
    beat_paths = sorted(Path('configs/train/beat/expression').glob('*.yaml'))
    hest1k_paths = sorted(Path('configs/train/hest1k/expression').glob('*/*.yaml'))

    assert sorted({path.name for path in hest1k_paths}) == [path.name for path in beat_paths]

    for path in beat_paths:
        cfg = Config.from_yaml(path)
        assert cfg.data.items_path == Path('default.json')
        assert cfg.data.metadata_path == Path('default/outer=0-inner=0-seed=0.parquet')
        assert cfg.data.panel_path == Path('default.yaml')

    for path in hest1k_paths:
        cfg = Config.from_yaml(path)
        organ = path.parent.name
        assert cfg.data.items_path == Path(f'{organ}.json')
        assert cfg.data.metadata_path == Path(f'{organ}/outer=0-inner=0-seed=0.parquet')
        assert cfg.data.panel_path == Path(f'hvg-{organ}-{organ}-outer=0-inner=0-seed=0.yaml')


def test_hest1k_organ_expression_configs_match_expected_variants_and_paths():
    base_variants = sorted(path.name for path in Path('configs/train/beat/expression').glob('*.yaml'))

    for organ in ['bowel', 'breast', 'lung', 'pancreas']:
        paths = sorted(Path('configs/train/hest1k/expression', organ).glob('*.yaml'))
        assert [path.name for path in paths] == base_variants

        for path in paths:
            cfg = Config.from_yaml(path)
            assert cfg.data.items_path == Path(f'{organ}.json')
            assert cfg.data.metadata_path == Path(f'{organ}/outer=0-inner=0-seed=0.parquet')
            assert cfg.data.panel_path == Path(f'hvg-{organ}-{organ}-outer=0-inner=0-seed=0.yaml')
            assert cfg.wandb.tags == [organ]


def test_train_configs_explicitly_set_learnable_gate_false():
    for path in Path('configs/train').rglob('*.yaml'):
        text = path.read_text()
        assert 'learnable_gate: false' in text, path

        cfg = Config.from_yaml(path)
        assert cfg.backbone.learnable_gate is False
