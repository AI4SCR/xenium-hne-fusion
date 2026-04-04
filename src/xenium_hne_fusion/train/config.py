from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class HeadConfig:
    hidden_dim: int = 32
    num_hidden_layers: int = 0
    dropout: float = 0.1


@dataclass
class BackboneConfig:
    morph_encoder_name: str | None = None
    morph_encoder_kws: dict[str, Any] | None = None
    expr_encoder_name: str | None = None
    expr_encoder_kws: dict[str, Any] | None = None
    fusion_strategy: Literal['concat', 'add'] | None = None
    fusion_stage: Literal['early', 'late'] | None = None
    global_pool: Literal['token', 'avg', 'max', 'flatten'] | None = None
    expr_token_pool: Literal['token', 'avg', 'max'] | None = None
    morph_token_pool: Literal['token', 'avg', 'max'] | None = None
    use_proj: bool = False
    use_modality_embed: bool = False
    learnable_gate: bool = False
    morph_key: str = 'image'
    expr_key: str = 'expr_tokens'
    allow_unimodal_routes: bool = False
    pos_embed_layer_name: str = '_pos_embed'
    freeze_morph_encoder: bool = False
    freeze_expr_encoder: bool = False


@dataclass
class DataConfig:
    name: str | None = None
    num_workers: int = 10
    batch_size: int = 256
    prefetch_factor: int | None = 4
    expr_pool: Literal['token', 'tile'] = 'token'
    panel_path: Path | None = None
    source_panel: list[str] | None = None
    target_panel: list[str] | None = None
    # Relative paths resolve under DATA_DIR/03_output/<name>/.
    items_path: Path | None = None
    metadata_path: Path | None = None
    cache_dir: Path | None = None


@dataclass
class LitConfig:
    target_key: str = 'target'
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-3
    eta: float = 1e-6
    schedule: Literal['cosine'] | None = 'cosine'
    num_warmup_epochs: int = 5


@dataclass
class TrainerConfig:
    max_epochs: int = 35
    max_time: str = '00:01:59:00'
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 1.0
    fast_dev_run: bool = False
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    limit_test_batches: float | int | None = None


@dataclass
class WandbConfig:
    project: str = ''
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    debug: bool = False
    head: HeadConfig = field(default_factory=HeadConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lit: LitConfig = field(default_factory=LitConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        import yaml
        data = yaml.safe_load(path.read_text()) or {}
        return _merge_dataclass(cls, data)


def _merge_dataclass(cls, data: dict):
    """Recursively construct a dataclass from a dict, using field defaults for missing keys."""
    import dataclasses
    import typing
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        hint = hints.get(f.name)
        if hint is not None and isinstance(hint, type) and dataclasses.is_dataclass(hint):
            val = _merge_dataclass(hint, val or {})
        elif hint is not None and _is_path_hint(hint) and val is not None:
            val = Path(val)
        elif val is not None and _is_scalar_hint(hint, float):
            val = float(val)
        elif val is not None and _is_scalar_hint(hint, int):
            val = int(val)
        kwargs[f.name] = val
    return cls(**kwargs)


def _is_path_hint(hint: Any) -> bool:
    import typing

    if hint is Path:
        return True
    return Path in typing.get_args(hint)


def _is_scalar_hint(hint: Any, scalar: type) -> bool:
    """Return True if hint resolves to scalar (or Optional[scalar])."""
    import typing

    if hint is scalar:
        return True
    return scalar in typing.get_args(hint) and type(None) in typing.get_args(hint)
