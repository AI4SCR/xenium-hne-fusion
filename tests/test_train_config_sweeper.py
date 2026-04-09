import importlib.util
import json
from pathlib import Path

import pytest

from xenium_hne_fusion.train.cli import parse_training_config
from xenium_hne_fusion.train.config import Config


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_training_config_applies_dotted_and_dict_overrides():
    cfg = parse_training_config(
        Path("configs/train/beat/expression/default/early-fusion.yaml"),
        overrides={
            "backbone.morph_encoder_name": "vit_base_patch16_224",
            "backbone.expr_encoder_kws": {
                "output_dim": 128,
                "hidden_dim": 128,
                "num_hidden_layers": 1,
                "dropout": 0.1,
            },
        },
    )

    assert cfg.backbone.morph_encoder_name == "vit_base_patch16_224"
    assert cfg.backbone.expr_encoder_kws == {
        "output_dim": 128,
        "hidden_dim": 128,
        "num_hidden_layers": 1,
        "dropout": 0.1,
    }


def test_sweeper_emits_only_matched_capacity_runs_for_multimodal_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_multimodal")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    manifest = module.main(Path("configs/train/beat/expression/default/early-fusion.yaml"), version="v1")

    assert len(manifest) == 2
    cfg_low = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__early-fusion/low.yaml")
    cfg_high = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__early-fusion/high.yaml")

    assert cfg_low.backbone.morph_encoder_name == "vit_small_patch16_224"
    assert cfg_low.backbone.expr_encoder_kws == {
        "output_dim": 32,
        "hidden_dim": 32,
        "num_hidden_layers": 1,
        "dropout": 0.1,
    }
    assert cfg_high.backbone.morph_encoder_name == "vit_base_patch16_224"
    assert cfg_high.backbone.expr_encoder_kws == {
        "output_dim": 128,
        "hidden_dim": 128,
        "num_hidden_layers": 1,
        "dropout": 0.1,
    }
    assert {row["wandb_name"] for row in manifest} == {
        "configs__train__beat__expression__default__early-fusion__low",
        "configs__train__beat__expression__default__early-fusion__high",
    }


def test_sweeper_ignores_expr_axis_for_vision_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_vision")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    manifest = module.main(Path("configs/train/beat/expression/default/vision.yaml"), version="v1")

    assert len(manifest) == 2
    cfg_low = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__vision/low.yaml")
    cfg_high = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__vision/high.yaml")

    assert cfg_low.backbone.expr_encoder_name is None
    assert cfg_low.backbone.expr_encoder_kws is None
    assert cfg_high.backbone.expr_encoder_name is None
    assert cfg_high.backbone.expr_encoder_kws is None
    assert {cfg_low.backbone.morph_encoder_name, cfg_high.backbone.morph_encoder_name} == {
        "vit_small_patch16_224",
        "vit_base_patch16_224",
    }


def test_sweeper_ignores_morph_axis_for_expr_only_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_expr")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    manifest = module.main(Path("configs/train/beat/expression/default/expr-token.yaml"), version="v1")

    assert len(manifest) == 2
    cfg_low = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__expr-token/low.yaml")
    cfg_high = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__expression__default__expr-token/high.yaml")

    assert cfg_low.backbone.morph_encoder_name is None
    assert cfg_high.backbone.morph_encoder_name is None
    assert cfg_low.backbone.expr_encoder_kws["output_dim"] == 32
    assert cfg_high.backbone.expr_encoder_kws["output_dim"] == 128
    assert all("command" in row for row in manifest)


def test_sweeper_round_trips_beat_cell_types_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_cell_types")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    manifest = module.main(Path("configs/train/beat/cell_types/early-fusion.yaml"), version="v1")

    assert len(manifest) == 2
    cfg = Config.from_yaml(tmp_path / "tasks/v1/configs__train__beat__cell-types__early-fusion/low.yaml")
    assert cfg.task.target == "cell_types"
    assert cfg.head.output_dim == 39


def test_sweeper_preserves_hest1k_tags_and_panel_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_hest1k")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    manifest = module.main(Path("configs/train/hest1k/expression/breast/early-fusion.yaml"), version="v1")

    assert len(manifest) == 2
    cfg = Config.from_yaml(tmp_path / "tasks/v1/configs__train__hest1k__expression__breast__early-fusion/high.yaml")
    assert cfg.data.name == "hest1k"
    assert cfg.data.panel_path == Path("hvg-breast-breast-outer=0-seed=0.yaml")
    assert cfg.wandb.tags == ["breast"]


def test_sweeper_writes_manifest_and_rejects_existing_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/launchers/train_config_sweeper.py", "train_config_sweeper_manifest")
    monkeypatch.setattr(module, "TASKS_ROOT", tmp_path / "tasks")

    module.main(Path("configs/train/beat/expression/default/early-fusion.yaml"), version="v1")
    manifest_path = tmp_path / "tasks/v1/configs__train__beat__expression__default__early-fusion/manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert len(manifest) == 2
    assert manifest[0]["version"] == "v1"

    with pytest.raises(AssertionError, match="Output directory already exists"):
        module.main(Path("configs/train/beat/expression/default/early-fusion.yaml"), version="v1")
