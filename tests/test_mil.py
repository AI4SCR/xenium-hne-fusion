from pathlib import Path

import pandas as pd
import pytest
import torch
from xenium_hne_fusion.train.config import Config, HeadConfig, TrainerConfig, WandbConfig
from xenium_hne_fusion.train.mil import (
    MILBagsDataset,
    build_mil_module,
    resolve_pretrained_run,
)
from xenium_hne_fusion.train.mil_config import (
    AggregatorConfig,
    MILConfig,
    MILDataConfig,
    MILLitConfig,
    MILTaskConfig,
    PretrainedConfig,
)
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws, build_supervised_lit
from xenium_hne_fusion.train.utils import prepare_training_config


# --- parser -------------------------------------------------------------------


def test_mil_parser_reads_yaml_config_into_milconfig(tmp_path: Path):
    config_path = tmp_path / "mil.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pretrained:",
                "  entity: chuv",
                "  project: mil-proj",
                "  run_id: run-123",
                "data:",
                "  name: beat",
                "aggregator:",
                "  name: attention",
                "  hidden_dim: 64",
                "  gated: true",
                "task:",
                "  kind: classification",
                "wandb:",
                "  project: mil-v0",
                "  name: mil-test",
            ]
        ),
        encoding="utf-8",
    )

    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(MILConfig, None)
    namespace = parser.parse_args(["--config", str(config_path)])
    init = parser.instantiate_classes(namespace)
    d = vars(init)
    d.pop("config", None)
    cfg = MILConfig(**d)

    assert isinstance(cfg, MILConfig)
    assert cfg.pretrained.run_id == "run-123"
    assert cfg.data.name == "beat"
    assert cfg.task.kind == "classification"
    assert cfg.aggregator.gated is True


# --- resolve_pretrained_run ---------------------------------------------------


def test_resolve_pretrained_run_reads_checkpoint_and_rebuilds_source_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint_path = tmp_path / "best.ckpt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")

    class FakeRun:
        config = {
            "best_model_path": str(checkpoint_path),
            "task": {"target": "expression"},
            "data": {
                "name": "beat",
                "items_path": "expr.json",
                "metadata_path": "expr/outer=0-inner=0-seed=0.parquet",
                "panel_path": "default.yaml",
            },
            "lit": {"target_key": "target"},
        }

    class FakeApi:
        def run(self, path: str):
            assert path == "chuv/mil-proj/run-123"
            return FakeRun()

    monkeypatch.setattr("xenium_hne_fusion.train.mil.wandb.Api", lambda: FakeApi())
    resolved = resolve_pretrained_run(PretrainedConfig(project="mil-proj", run_id="run-123"))

    assert resolved.checkpoint_path == checkpoint_path
    assert resolved.source_config.task.target == "expression"
    assert resolved.source_config.data.name == "beat"
    assert resolved.source_config.data.metadata_path == Path("expr/outer=0-inner=0-seed=0.parquet")


def test_resolve_pretrained_run_requires_best_model_path(monkeypatch: pytest.MonkeyPatch):
    class FakeRun:
        config = {"task": {"target": "expression"}}

    class FakeApi:
        def run(self, path: str):
            return FakeRun()

    monkeypatch.setattr("xenium_hne_fusion.train.mil.wandb.Api", lambda: FakeApi())

    with pytest.raises(AssertionError, match="best_model_path"):
        resolve_pretrained_run(PretrainedConfig(project="mil-proj", run_id="run-123"))



# --- build_supervised_builders ------------------------------------------------


def test_build_supervised_builders_split_model_and_dataset_kws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cfg = Config()
    cfg.task.target = "expression"
    cfg.lit.target_key = "target"
    cfg.data.name = "beat"
    cfg.data.items_path = tmp_path / "items.json"
    cfg.data.metadata_path = tmp_path / "metadata.parquet"
    cfg.data.panel_path = tmp_path / "panel.yaml"
    cfg.data.source_panel = ["A", "B"]
    cfg.data.target_panel = ["C"]
    cfg.backbone.expr_encoder_name = "mlp"
    cfg.backbone.expr_encoder_kws = {"hidden_dim": 4, "output_dim": 4}
    cfg.data.items_path.write_text("[]", encoding="utf-8")
    pd.DataFrame([{"sample_id": "S1", "split": "fit"}]).set_index("sample_id").to_parquet(cfg.data.metadata_path)
    cfg.data.panel_path.write_text("{}", encoding="utf-8")

    resolved = prepare_training_config(cfg)
    lit = build_supervised_lit(resolved)
    dataset_kws = build_supervised_dataset_kws(resolved)

    assert lit.num_outputs == 1
    assert dataset_kws["id_key"] == "id"
    assert dataset_kws["items_path"] == cfg.data.items_path