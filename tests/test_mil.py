from pathlib import Path

import pandas as pd
import pytest
import torch
from ai4bmr_learn.data.splits import Split

from xenium_hne_fusion.train.config import Config, HeadConfig, TrainerConfig, WandbConfig
from xenium_hne_fusion.train.mil import (
    MILBagsDataset,
    build_mil_module,
    build_sample_level_mil_metadata,
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


# --- build_sample_level_mil_metadata ------------------------------------------


def test_build_sample_level_mil_metadata_collapses_tile_rows(tmp_path: Path):
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "tile_id": 0,
                "tile_dir": "/tmp/a",
                Split.COLUMN_NAME.value: Split.FIT.value,
                "response": 1.5,
                "site": "A",
            },
            {
                "sample_id": "S1",
                "tile_id": 1,
                "tile_dir": "/tmp/b",
                Split.COLUMN_NAME.value: Split.FIT.value,
                "response": 1.5,
                "site": "A",
            },
            {
                "sample_id": "S2",
                "tile_id": 0,
                "tile_dir": "/tmp/c",
                Split.COLUMN_NAME.value: Split.TEST.value,
                "response": 2.5,
                "site": "B",
            },
        ]
    ).to_parquet(split_path, index=False)

    output_path = build_sample_level_mil_metadata(
        metadata_path=split_path,
        target_key="metadata.response",
        task_kind="regression",
        output_path=tmp_path / "sample.parquet",
    )
    result = pd.read_parquet(output_path)

    assert list(result.index.astype(str)) == ["S1", "S2"]
    assert result.loc["S1", Split.COLUMN_NAME.value] == Split.FIT.value
    assert result.loc["S1", "response"] == pytest.approx(1.5)
    assert "tile_id" not in result.columns
    assert "tile_dir" not in result.columns


def test_build_sample_level_mil_metadata_rejects_inconsistent_sample_values(tmp_path: Path):
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "label": "A"},
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.TEST.value, "label": "A"},
        ]
    ).to_parquet(split_path, index=False)

    with pytest.raises(AssertionError, match="inconsistent split"):
        build_sample_level_mil_metadata(
            metadata_path=split_path,
            target_key="metadata.label",
            task_kind="classification",
            output_path=tmp_path / "sample.parquet",
        )


def test_build_sample_level_mil_metadata_converts_classification_target_in_place(tmp_path: Path):
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "label": "A"},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.TEST.value, "label": "B"},
        ]
    ).to_parquet(split_path, index=False)

    output_path = build_sample_level_mil_metadata(
        metadata_path=split_path,
        target_key="metadata.label",
        task_kind="classification",
        output_path=tmp_path / "sample.parquet",
    )
    result = pd.read_parquet(output_path)

    assert result.loc["S1", "label"] == 0
    assert result.loc["S2", "label"] == 1


def test_build_sample_level_mil_metadata_joins_clinical_path(tmp_path: Path):
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.TEST.value},
        ]
    ).to_parquet(split_path, index=False)

    clinical_path = tmp_path / "clinical.parquet"
    pd.DataFrame(
        [{"sample_id": "S1", "grade": 1.0}, {"sample_id": "S2", "grade": 2.0}]
    ).to_parquet(clinical_path, index=False)

    output_path = build_sample_level_mil_metadata(
        metadata_path=split_path,
        target_key="metadata.grade",
        task_kind="regression",
        output_path=tmp_path / "sample.parquet",
        clinical_path=clinical_path,
    )
    result = pd.read_parquet(output_path)

    assert result.loc["S1", "grade"] == pytest.approx(1.0)
    assert result.loc["S2", "grade"] == pytest.approx(2.0)


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

    lit = build_supervised_lit(cfg)
    dataset_kws = build_supervised_dataset_kws(cfg)

    assert lit.num_outputs == 1
    assert dataset_kws["id_key"] == "id"
    assert dataset_kws["items_path"] == cfg.data.items_path