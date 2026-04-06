from pathlib import Path

import pytest

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.utils import resolve_training_paths
from xenium_hne_fusion.utils.getters import get_panels_dir


def test_resolve_training_paths_defaults_to_dataset_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'metadata.parquet'
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('XHF_REPO_ROOT', str(tmp_path))

    cfg = Config()
    cfg.data.name = 'hest1k'
    cfg.data.metadata_path = metadata_path

    cfg, output_dir = resolve_training_paths(cfg)

    assert output_dir == (data_dir / '03_output' / 'hest1k').resolve()
    assert cfg.data.items_path == output_dir / 'items' / 'all.json'
    assert cfg.data.cache_dir == output_dir / 'cache'
    assert cfg.data.metadata_path == metadata_path


def test_resolve_training_paths_resolves_relative_paths_under_dataset_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('XHF_REPO_ROOT', str(tmp_path))

    cfg = Config()
    cfg.data.name = 'beat'
    cfg.data.items_path = Path('items/train.json')
    cfg.data.metadata_path = Path('default/outer=0-inner=0-seed=0.parquet')
    cfg.data.panel_path = Path('default.yaml')
    cfg.data.cache_dir = Path('cache/run-a')

    cfg, output_dir = resolve_training_paths(cfg)

    assert cfg.data.items_path == output_dir / 'items/train.json'
    assert cfg.data.metadata_path == output_dir / 'splits' / 'default/outer=0-inner=0-seed=0.parquet'
    assert cfg.data.panel_path == get_panels_dir('beat') / 'default.yaml'
    assert cfg.data.cache_dir == output_dir / 'cache/run-a'


def test_resolve_training_paths_keeps_absolute_metadata_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'custom' / 'split.parquet'
    monkeypatch.setenv('DATA_DIR', str(data_dir))
    monkeypatch.setenv('XHF_REPO_ROOT', str(tmp_path))

    cfg = Config()
    cfg.data.name = 'beat'
    cfg.data.metadata_path = metadata_path

    cfg, _ = resolve_training_paths(cfg)

    assert cfg.data.metadata_path == metadata_path


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
        assert isinstance(cfg.data.metadata_path, Path)
        assert isinstance(cfg.data.panel_path, Path)
        assert cfg.head.num_hidden_layers == 0
        assert cfg.lit.target_key is not None
        assert cfg.wandb.project == 'debug'
        assert cfg.trainer.max_time == '00:01:45:00'
        assert cfg.trainer.max_epochs == 35
        assert cfg.trainer.gradient_clip_val == 1.0


def test_train_configs_use_expected_panel_defaults():
    beat_cfg = Config.from_yaml(Path('configs/train/beat/expression.yaml'))
    hest1k_cfg = Config.from_yaml(Path('configs/train/hest1k/expression.yaml'))

    assert beat_cfg.data.panel_path == Path('default.yaml')
    assert hest1k_cfg.data.panel_path == Path('hvg-default-default-outer=0-inner=0-seed=0.yaml')
