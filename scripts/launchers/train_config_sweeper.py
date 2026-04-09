from __future__ import annotations

import json
import re
import shutil
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import auto_cli

from xenium_hne_fusion.train.cli import config_to_document, parse_training_config
from xenium_hne_fusion.train.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"

LOW_MORPH_ENCODER = "vit_small_patch16_224"
HIGH_MORPH_ENCODER = "vit_base_patch16_224"
LOW_EXPR_ENCODER_KWS = {
    "output_dim": 32,
    "hidden_dim": 32,
    "num_hidden_layers": 1,
    "dropout": 0.1,
}
HIGH_EXPR_ENCODER_KWS = {
    "output_dim": 128,
    "hidden_dim": 128,
    "num_hidden_layers": 1,
    "dropout": 0.1,
}

SWEEP_PARAMETERS: dict[str, list[Any]] = {
    "backbone.morph_encoder_name": [LOW_MORPH_ENCODER, HIGH_MORPH_ENCODER],
    "backbone.expr_encoder_kws": [LOW_EXPR_ENCODER_KWS, HIGH_EXPR_ENCODER_KWS],
}


def expand_sweep(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters)
    values = [parameters[key] for key in keys]
    return [dict(zip(keys, candidate, strict=True)) for candidate in product(*values)]


def filter_candidate(base_cfg: Config, overrides: dict[str, Any]) -> bool:
    has_morph = base_cfg.backbone.morph_encoder_name is not None
    has_expr = base_cfg.backbone.expr_encoder_name is not None
    if not (has_morph and has_expr):
        return True

    morph_name = overrides.get("backbone.morph_encoder_name", base_cfg.backbone.morph_encoder_name)
    expr_encoder_kws = overrides.get("backbone.expr_encoder_kws", base_cfg.backbone.expr_encoder_kws)
    morph_capacity = _classify_morph_capacity(morph_name)
    expr_capacity = _classify_expr_capacity(expr_encoder_kws)
    if morph_capacity is None or expr_capacity is None:
        return True
    return morph_capacity == expr_capacity


def apply_overrides(base_config_path: Path, overrides: dict[str, Any]) -> Config:
    return parse_training_config(base_config_path, overrides=overrides)


def build_run_name(base_cfg: Config, overrides: dict[str, Any], config_path: Path) -> str:
    morph_name = overrides.get("backbone.morph_encoder_name", base_cfg.backbone.morph_encoder_name)
    expr_encoder_kws = overrides.get("backbone.expr_encoder_kws", base_cfg.backbone.expr_encoder_kws)

    morph_capacity = _classify_morph_capacity(morph_name)
    expr_capacity = _classify_expr_capacity(expr_encoder_kws)
    if morph_capacity is not None and morph_capacity == expr_capacity:
        return morph_capacity
    if morph_capacity is not None and base_cfg.backbone.expr_encoder_name is None:
        return morph_capacity
    if expr_capacity is not None and base_cfg.backbone.morph_encoder_name is None:
        return expr_capacity

    labels = [_format_override_label(key, value) for key, value in overrides.items()]
    joined = "__".join(labels)
    return _slugify(joined) if joined else _slugify(config_path.stem)


def write_config(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_document(cfg), sort_keys=False))


def validate_written_config(path: Path) -> Config:
    return Config.from_yaml(path)


def main(config: Path, version: str, overwrite: bool = False) -> list[dict[str, Any]]:
    base_config_path = config.resolve()
    assert base_config_path.exists(), f"Base config not found: {base_config_path}"

    base_cfg = Config.from_yaml(base_config_path)
    output_dir = TASKS_ROOT / version / flatten_config_stem(base_config_path)
    if output_dir.exists():
        assert overwrite, f"Output directory already exists: {output_dir}"
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for overrides in _resolve_candidates(base_cfg):
        cfg = apply_overrides(base_config_path, overrides)
        run_stem = build_run_name(base_cfg, overrides, base_config_path)
        cfg.wandb.name = f"{flatten_config_stem(base_config_path)}__{run_stem}"
        cfg.wandb.group = version

        config_path = output_dir / f"{run_stem}.yaml"
        write_config(cfg, config_path)
        validate_written_config(config_path)

        relative_config_path = _relative_to_repo(config_path)
        manifest_rows.append(
            {
                "version": version,
                "base_config_path": str(_relative_to_repo(base_config_path)),
                "generated_config_path": str(relative_config_path),
                "overrides": _json_ready(overrides),
                "wandb_name": cfg.wandb.name,
                "command": f"uv run scripts/train/supervised.py --config {relative_config_path}",
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(manifest_rows)} configs to {output_dir}")
    return manifest_rows


def _resolve_candidates(base_cfg: Config) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for overrides in expand_sweep(SWEEP_PARAMETERS):
        normalized = _prune_irrelevant_overrides(base_cfg, overrides)
        if not filter_candidate(base_cfg, normalized):
            continue
        key = json.dumps(_json_ready(normalized), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _prune_irrelevant_overrides(base_cfg: Config, overrides: dict[str, Any]) -> dict[str, Any]:
    has_morph = base_cfg.backbone.morph_encoder_name is not None
    has_expr = base_cfg.backbone.expr_encoder_name is not None

    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        if key.startswith("backbone.morph_") and not has_morph:
            continue
        if key.startswith("backbone.expr_") and not has_expr:
            continue
        normalized[key] = value
    return normalized


def flatten_config_stem(path: Path) -> str:
    relative = _relative_to_repo(path)
    parts = list(relative.parts)
    parts[-1] = Path(parts[-1]).stem
    return "__".join(_slugify(part) for part in parts)


def _relative_to_repo(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path.resolve()


def _classify_morph_capacity(name: str | None) -> str | None:
    if name == LOW_MORPH_ENCODER:
        return "low"
    if name == HIGH_MORPH_ENCODER:
        return "high"
    return None


def _classify_expr_capacity(kws: dict[str, Any] | None) -> str | None:
    if kws == LOW_EXPR_ENCODER_KWS:
        return "low"
    if kws == HIGH_EXPR_ENCODER_KWS:
        return "high"
    return None


def _format_override_label(key: str, value: Any) -> str:
    label = key.split(".")[-1]
    if isinstance(value, dict):
        inner = "-".join(f"{_slugify(inner_key)}-{_slugify(inner_value)}" for inner_key, inner_value in value.items())
        return f"{_slugify(label)}-{inner}"
    return f"{_slugify(label)}-{_slugify(value)}"


def _slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    auto_cli(main)
