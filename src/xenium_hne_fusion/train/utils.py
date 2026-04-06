from pathlib import Path

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.utils.getters import get_managed_paths, get_panels_dir


def set_fast_dev_run_settings(cfg: Config) -> Config:
    cfg.wandb.project = 'debug'
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.trainer.max_epochs = 3
    cfg.trainer.limit_train_batches = 2
    cfg.trainer.limit_val_batches = 2
    cfg.trainer.limit_test_batches = 2
    cfg.lit.num_warmup_epochs = 2
    return cfg


def resolve_training_paths(cfg: Config) -> tuple[Config, Path]:
    name = cfg.data.name
    assert name is not None, 'cfg.data.name must be set'

    output_dir = get_managed_paths(name).output_dir
    panels_dir = get_panels_dir(name)
    cfg.data.items_path = _resolve_path(cfg.data.items_path, root=output_dir, default=output_dir / 'items' / 'all.json')
    cfg.data.metadata_path = _resolve_path(cfg.data.metadata_path, root=output_dir)
    cfg.data.panel_path = _resolve_path(cfg.data.panel_path, root=panels_dir)
    cfg.data.cache_dir = _resolve_path(cfg.data.cache_dir, root=output_dir, default=output_dir / 'cache')

    assert cfg.data.items_path is not None, 'cfg.data.items_path must be set'
    assert cfg.data.metadata_path is not None, 'cfg.data.metadata_path must be set'
    assert cfg.data.cache_dir is not None, 'cfg.data.cache_dir must be set'
    return cfg, output_dir


def _resolve_path(path: Path | None, *, root: Path | None = None, default: Path | None = None) -> Path | None:
    if path is None:
        return default
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    return path.resolve()
