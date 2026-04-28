import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.datasets import pad_bags_collate, write_mil_items_from_cache
from ai4bmr_learn.models.mil import AttentionAggregation

from xenium_hne_fusion.train.config import HeadConfig, TrainerConfig, WandbConfig
from xenium_hne_fusion.train.mil import (
    MetadataBagsDataset,
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


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mil_parser_reads_yaml_config_into_namespace(tmp_path: Path):
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
                "  items_path: expr.json",
                "  split_metadata_path: expr/outer=0-inner=0-seed=0.parquet",
                "  target_column: response",
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

    module = _load_script("scripts/train/mil.py", "train_mil_script")
    parser = module._build_parser()
    namespace = parser.parse_args(["--config", str(config_path)])
    data = namespace.as_dict()

    assert data["pretrained"]["run_id"] == "run-123"
    assert data["data"]["items_path"] == Path("expr.json")
    assert data["data"]["split_metadata_path"] == Path("expr/outer=0-inner=0-seed=0.parquet")
    assert data["data"]["target_column"] == "response"
    assert data["task"]["kind"] == "classification"
    assert data["aggregator"]["gated"] is True


def test_mil_namespace_bridge_returns_concrete_config(tmp_path: Path):
    config_path = tmp_path / "mil.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pretrained:",
                "  project: mil-proj",
                "  run_id: run-123",
                "data:",
                "  name: beat",
                "  items_path: expr.json",
                "  split_metadata_path: expr/outer=0-inner=0-seed=0.parquet",
                "  target_column: score",
                "wandb:",
                "  project: mil-v0",
            ]
        ),
        encoding="utf-8",
    )

    module = _load_script("scripts/train/mil.py", "train_mil_bridge_script")
    parser = module._build_parser()
    namespace = parser.parse_args(["--config", str(config_path)])
    cfg = module._namespace_to_config(namespace)

    assert isinstance(cfg, MILConfig)
    assert isinstance(cfg.pretrained, PretrainedConfig)
    assert isinstance(cfg.data, MILDataConfig)
    assert isinstance(cfg.task, MILTaskConfig)
    assert isinstance(cfg.aggregator, AggregatorConfig)
    assert isinstance(cfg.head, HeadConfig)
    assert isinstance(cfg.lit, MILLitConfig)
    assert isinstance(cfg.trainer, TrainerConfig)
    assert isinstance(cfg.wandb, WandbConfig)
    assert cfg.data.target_column == "score"


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
        split_metadata_path=split_path,
        target_column="response",
        task_kind="regression",
        output_path=tmp_path / "sample.parquet",
    )
    result = pd.read_parquet(output_path)

    assert list(result.index.astype(str)) == ["S1", "S2"]
    assert result.loc["S1", Split.COLUMN_NAME.value] == Split.FIT.value
    assert result.loc["S1", "response"] == pytest.approx(1.5)
    assert result.loc["S1", "target"] == pytest.approx(1.5)
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
            split_metadata_path=split_path,
            target_column="label",
            task_kind="classification",
            output_path=tmp_path / "sample.parquet",
        )


def test_resolve_pretrained_run_reads_checkpoint_and_rebuilds_source_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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


def test_resolve_pretrained_run_requires_best_model_path(monkeypatch: pytest.MonkeyPatch):
    class FakeRun:
        config = {"task": {"target": "expression"}}

    class FakeApi:
        def run(self, path: str):
            return FakeRun()

    monkeypatch.setattr("xenium_hne_fusion.train.mil.wandb.Api", lambda: FakeApi())

    with pytest.raises(AssertionError, match="best_model_path"):
        resolve_pretrained_run(PretrainedConfig(project="mil-proj", run_id="run-123"))


def test_write_mil_items_from_cache_and_dataset_module_smoke(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "prediction"
    cache_dir.mkdir(parents=True)
    torch.save(
        [
            {
                "sample_id": ["S1", "S1", "S2"],
                "z": torch.tensor([[1.0, 0.0], [0.5, 1.5], [2.0, 3.0]]),
            }
        ],
        cache_dir / "000000.pt",
    )

    items_path = write_mil_items_from_cache(
        cache_dir=cache_dir,
        items_path=tmp_path / "mil-items.json",
        id_key="sample_id",
    )
    items = json.loads(items_path.read_text(encoding="utf-8"))
    assert [item["sample_id"] for item in items] == ["S1", "S1", "S2"]

    metadata_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "target": 1.0},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.FIT.value, "target": 0.0},
        ]
    ).set_index("sample_id").to_parquet(metadata_path)

    dataset = MetadataBagsDataset(
        items_path=items_path,
        metadata_path=metadata_path,
        split=Split.FIT.value,
        task_kind="regression",
    )
    dataset.setup()

    first = dataset[0]
    assert first["bag"].shape == (2, 2)
    assert first["target"].dtype == torch.float32

    batch = pad_bags_collate([dataset[0], dataset[1]])
    cfg = MILConfig()
    cfg.task.kind = "regression"
    cfg.aggregator.name = "attention"
    cfg.aggregator.hidden_dim = 4
    module = build_mil_module(cfg=cfg, input_dim=2)
    module.log = lambda *args, **kwargs: None
    output = module.training_step(batch, 0)

    assert torch.isfinite(output["loss"])
    assert output["embedding"].shape == (2, 2)
    assert output["weights"].shape == (2, 2)


def test_classification_bags_dataset_and_module_smoke(tmp_path: Path):
    items_path = tmp_path / "items.json"
    items_path.write_text(
        json.dumps(
            [
                {"id": "0", "sample_id": "S1", "z": [1.0, 2.0]},
                {"id": "1", "sample_id": "S2", "z": [3.0, 4.0]},
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "target": 0},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.FIT.value, "target": 1},
        ]
    ).set_index("sample_id").to_parquet(metadata_path)

    dataset = MetadataBagsDataset(
        items_path=items_path,
        metadata_path=metadata_path,
        split=Split.FIT.value,
        task_kind="classification",
    )
    dataset.setup()
    batch = pad_bags_collate([dataset[0], dataset[1]])

    cfg = MILConfig(
        task=MILTaskConfig(kind="classification"),
        aggregator=AggregatorConfig(name="attention", hidden_dim=4, gated=True),
    )
    module = build_mil_module(cfg=cfg, input_dim=2, num_classes=2)
    module.log = lambda *args, **kwargs: None

    assert isinstance(module.aggregator, AttentionAggregation)
    output = module.training_step(batch, 0)
    assert torch.isfinite(output["loss"])
    assert output["prediction"].dtype == torch.long
