import os
from pathlib import Path

import yaml

from xenium_hne_fusion.train.config import TrainingConfig
from xenium_hne_fusion.utils.getters import ManagedPaths


def resolve_training_paths(cfg: TrainingConfig) -> TrainingConfig:
    """Resolve items/metadata/panel/cache paths and set `cfg.output_dir`. Mutates `cfg` in place."""
    assert cfg.data.data_dir is not None, 'cfg.data.data_dir must be set'
    assert cfg.data.name is not None, 'cfg.data.name must be set'
    assert cfg.data.items_path is not None, 'cfg.data.items_path must be set'
    assert cfg.data.metadata_path is not None, 'cfg.data.metadata_path must be set'
    assert cfg.data.panel_path is not None, 'cfg.data.panel_path must be set'

    managed = ManagedPaths(data_dir=cfg.data.data_dir, name=cfg.data.name)
    cfg.output_dir = managed.output_dir
    cfg.data.items_path = _resolve_path(cfg.data.items_path, root=managed.output_dir / 'items')
    cfg.data.metadata_path = _resolve_path(cfg.data.metadata_path, root=managed.output_dir / 'splits')
    cfg.data.panel_path = _resolve_path(cfg.data.panel_path, root=managed.panels_dir)
    cfg.data.cache_dir = _resolve_path(cfg.data.cache_dir, root=managed.output_dir / 'cache')
    return cfg


def load_panel_config(cfg: TrainingConfig) -> TrainingConfig:
    if cfg.data.panel_path is None:
        return cfg
    assert cfg.data.panel_path.exists(), f"Panel file not found: {cfg.data.panel_path}"
    panel = yaml.safe_load(cfg.data.panel_path.read_text()) or {}
    if cfg.data.source_panel is None:
        cfg.data.source_panel = panel.get("source_panel")
    if cfg.data.target_panel is None:
        cfg.data.target_panel = panel.get("target_panel")
    return cfg


def validate_task_config(cfg: TrainingConfig) -> None:
    assert cfg.task.target is not None, "cfg.task.target"
    assert cfg.lit.target_key is not None, "cfg.lit.target_key must be set explicitly"

    if cfg.task.target == "expression":
        assert cfg.head.output_dim is None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "target", "cfg.lit.target_key"
        assert cfg.data.target_panel is not None, "cfg.data.target_panel"
        if cfg.data.source_panel is not None:
            assert set(cfg.data.source_panel).isdisjoint(set(cfg.data.target_panel))
        return

    if cfg.task.target == "cell_types":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "target", "cfg.lit.target_key"
        assert cfg.data.cell_type_col is not None, "cfg.data.cell_type_col"
        return

    if cfg.task.target == "proteins":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "proteins", "cfg.lit.target_key"
        return

    if cfg.task.target == "rgb":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "rgb", f"cfg.lit.target_key is {cfg.lit.target_key}"
        return

    if cfg.task.target == "conch_class":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "conch_class", f"cfg.lit.target_key is {cfg.lit.target_key}"
        return

    if cfg.task.target == "conch_scores":
        assert cfg.head.output_dim is not None, "cfg.head.output_dim"
        assert cfg.lit.target_key == "conch_scores", f"cfg.lit.target_key is {cfg.lit.target_key}"
        return

    raise ValueError(f"Unknown task target: {cfg.task.target}")


def resolve_num_source_genes(cfg: TrainingConfig) -> int | None:
    if cfg.backbone.expr_encoder_name is None:
        return None
    assert cfg.data.source_panel is not None, "cfg.data.source_panel must be set when using an expression encoder"
    return len(cfg.data.source_panel)


def resolve_num_outputs(cfg: TrainingConfig) -> int:
    if cfg.task.target == "expression":
        assert cfg.data.target_panel is not None
        return len(cfg.data.target_panel)
    assert cfg.head.output_dim is not None
    return cfg.head.output_dim


def resolve_training_config(cfg: TrainingConfig) -> TrainingConfig:
    """Single entrypoint for config resolution: paths, panel, task validation, and derived dims.

    Mutates and returns `cfg` with `output_dir`, `num_source_genes`, and `num_outputs` populated.
    """
    resolve_training_paths(cfg)
    load_panel_config(cfg)
    validate_task_config(cfg)
    cfg.num_source_genes = resolve_num_source_genes(cfg)
    cfg.num_outputs = resolve_num_outputs(cfg)
    return cfg


def infer_head_input_dim(
    *,
    fusion_stage: str | None,
    fusion_strategy: str | None,
    morph_encoder_dim: int | None,
    expr_encoder_dim: int | None,
) -> int:
    if fusion_stage == "late" and fusion_strategy == "concat":
        assert morph_encoder_dim is not None, "morph_encoder_dim must be set for late concat fusion"
        return morph_encoder_dim * 2
    embed_dim = morph_encoder_dim or expr_encoder_dim
    assert embed_dim is not None, "Could not infer head input dim"
    return embed_dim


def set_fast_dev_run_settings(cfg: TrainingConfig) -> TrainingConfig:
    cfg.wandb.project = 'debug'
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.trainer.max_epochs = 3
    cfg.trainer.limit_train_batches = 2
    cfg.trainer.limit_val_batches = 2
    cfg.trainer.limit_test_batches = 2
    cfg.trainer.limit_predict_batches = 2
    cfg.lit.num_warmup_epochs = 2
    return cfg


def _resolve_path(path: Path | None, *, root: Path | None = None, default: Path | None = None) -> Path | None:
    if path is None:
        return default
    path = Path(os.path.expandvars(path))  # expand $TMPDIR etc. at runtime
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    return path.resolve()
