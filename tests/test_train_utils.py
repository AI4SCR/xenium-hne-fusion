from pathlib import Path

import pytest

from xenium_hne_fusion.train.config import TrainingConfig
from xenium_hne_fusion.train.utils import resolve_training_paths


def test_resolve_training_paths_requires_explicit_training_data_references(tmp_path: Path):
    cfg = TrainingConfig()
    cfg.data.data_dir = tmp_path / 'data'
    cfg.data.name = 'hest1k'

    with pytest.raises(AssertionError, match='cfg.data.items_path'):
        resolve_training_paths(cfg)


def test_resolve_training_paths_keeps_unset_cache_dir_disabled(tmp_path: Path):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'metadata.parquet'

    cfg = TrainingConfig()
    cfg.data.data_dir = data_dir
    cfg.data.name = 'hest1k'
    cfg.data.items_path = Path('all.json')
    cfg.data.metadata_path = metadata_path
    cfg.data.panel_path = Path('hvg-default-default-outer=0-seed=0.yaml')

    cfg = resolve_training_paths(cfg)

    assert cfg.output_dir == data_dir / '03_output' / 'hest1k'
    assert cfg.data.items_path == cfg.output_dir / 'items/all.json'
    assert cfg.data.cache_dir is None
    assert cfg.data.metadata_path == metadata_path
    assert cfg.data.panel_path == cfg.output_dir / 'panels/hvg-default-default-outer=0-seed=0.yaml'


def test_resolve_training_paths_resolves_relative_paths_under_dataset_output_root(tmp_path: Path):
    data_dir = tmp_path / 'data'

    cfg = TrainingConfig()
    cfg.data.data_dir = data_dir
    cfg.data.name = 'beat'
    cfg.data.items_path = Path('train.json')
    cfg.data.metadata_path = Path('default/outer=0-seed=0.parquet')
    cfg.data.panel_path = Path('expr.yaml')
    cfg.data.cache_dir = Path('run-a')

    cfg = resolve_training_paths(cfg)

    assert cfg.data.items_path == cfg.output_dir / 'items/train.json'
    assert cfg.data.metadata_path == cfg.output_dir / 'splits' / 'default/outer=0-seed=0.parquet'
    assert cfg.data.panel_path == cfg.output_dir / 'panels/expr.yaml'
    assert cfg.data.cache_dir == cfg.output_dir / 'cache/run-a'


def test_resolve_training_paths_keeps_absolute_paths(tmp_path: Path):
    data_dir = tmp_path / 'data'
    metadata_path = tmp_path / 'custom' / 'split.parquet'
    items_path = tmp_path / 'custom' / 'items.json'
    panel_path = tmp_path / 'custom' / 'panel.yaml'
    cache_dir = tmp_path / 'custom' / 'cache'

    cfg = TrainingConfig()
    cfg.data.data_dir = data_dir
    cfg.data.name = 'beat'
    cfg.data.metadata_path = metadata_path
    cfg.data.items_path = items_path
    cfg.data.panel_path = panel_path
    cfg.data.cache_dir = cache_dir

    cfg = resolve_training_paths(cfg)

    assert cfg.data.items_path == items_path
    assert cfg.data.metadata_path == metadata_path
    assert cfg.data.panel_path == panel_path
    assert cfg.data.cache_dir == cache_dir


def test_resolve_training_paths_requires_name(tmp_path: Path):
    cfg = TrainingConfig()
    cfg.data.data_dir = tmp_path / 'data'
    cfg.data.metadata_path = tmp_path / 'metadata.parquet'

    with pytest.raises(AssertionError, match='cfg.data.name'):
        resolve_training_paths(cfg)


def test_resolve_training_paths_requires_data_dir(tmp_path: Path):
    cfg = TrainingConfig()
    cfg.data.name = 'beat'
    cfg.data.metadata_path = tmp_path / 'metadata.parquet'

    with pytest.raises(AssertionError, match='cfg.data.data_dir'):
        resolve_training_paths(cfg)


def test_train_configs_load_explicit_data_head_wandb_and_trainer_fields():
    for path in Path('configs/train').rglob('*.yaml'):
        cfg = TrainingConfig.from_yaml(path)
        assert cfg.task.target is not None
        assert cfg.data.name is not None
        assert isinstance(cfg.data.data_dir, Path)
        assert isinstance(cfg.data.items_path, Path)
        assert isinstance(cfg.data.metadata_path, Path)
        assert isinstance(cfg.data.panel_path, Path)
        assert cfg.head.num_hidden_layers == 0
        assert cfg.lit.target_key is not None
        assert cfg.wandb.project is not None
        assert cfg.wandb.name == path.stem


def test_train_configs_explicitly_set_learnable_gate_false():
    for path in Path('configs/train').rglob('*.yaml'):
        text = path.read_text()
        assert 'learnable_gate: false' in text, path

        cfg = TrainingConfig.from_yaml(path)
        assert cfg.backbone.learnable_gate is False
