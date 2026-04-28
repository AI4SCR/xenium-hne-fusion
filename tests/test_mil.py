import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch
from ai4bmr_learn.data.splits import Split
from ai4bmr_learn.datasets import pad_bags_collate, write_mil_items_from_cache
from ai4bmr_learn.models.mil import AttentionAggregation

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.config import HeadConfig, TrainerConfig, WandbConfig
from xenium_hne_fusion.train.mil import (
    MetadataBagsDataset,
    build_mil_module,
    build_sample_level_mil_metadata,
    extract_mil_embeddings,
    resolve_pretrained_run,
    write_mil_items_from_prediction_cache,
)
from xenium_hne_fusion.train.supervised import build_supervised_lit
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
                "  metadata_path: expr/outer=0-inner=0-seed=0.parquet",
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
    assert data["data"]["metadata_path"] == Path("expr/outer=0-inner=0-seed=0.parquet")
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
                "  metadata_path: expr/outer=0-inner=0-seed=0.parquet",
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
    assert cfg.data.metadata_path == Path("expr/outer=0-inner=0-seed=0.parquet")


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
                "id": ["tile-0", "tile-1", "tile-2"],
                "sample_id": ["S1", "S1", "S2"],
                "z": torch.tensor([[1.0, 0.0], [0.5, 1.5], [2.0, 3.0]]),
            }
        ],
        cache_dir / "000000.pt",
    )

    items_path = write_mil_items_from_prediction_cache(cache_dir=cache_dir, items_path=tmp_path / "mil-items.json")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    assert [item["sample_id"] for item in items] == ["S1", "S2"]
    assert all("instance_ids" in item for item in items)
    assert all("z_path" in item for item in items)

    metadata_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "response": 1.0},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.FIT.value, "response": 0.0},
        ]
    ).set_index("sample_id").to_parquet(metadata_path)

    dataset = MetadataBagsDataset(
        items_path=items_path,
        metadata_path=metadata_path,
        split=Split.FIT.value,
        task_kind="regression",
        target_key="metadata.response",
    )
    dataset.setup()

    first = dataset[0]
    assert first["bag"].shape == (2, 2)
    assert first["metadata"]["response"] == pytest.approx(1.0)

    batch = pad_bags_collate([dataset[0], dataset[1]])
    cfg = MILConfig()
    cfg.task.kind = "regression"
    cfg.aggregator.name = "attention"
    cfg.aggregator.hidden_dim = 4
    cfg.lit.target_key = "metadata.response"
    module = build_mil_module(cfg=cfg, input_dim=2)
    module.log = lambda *args, **kwargs: None
    output = module.training_step(batch, 0)

    assert torch.isfinite(output["loss"])
    assert output["embedding"].shape == (2, 2)
    assert output["weights"].shape == (2, 2)


def test_classification_bags_dataset_and_module_smoke(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "prediction"
    cache_dir.mkdir(parents=True)
    torch.save(
        [{"id": ["tile-0", "tile-1"], "sample_id": ["S1", "S2"], "z": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}],
        cache_dir / "000000.pt",
    )
    items_path = write_mil_items_from_prediction_cache(cache_dir=cache_dir, items_path=tmp_path / "items.json")
    metadata_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        [
            {"sample_id": "S1", Split.COLUMN_NAME.value: Split.FIT.value, "label": 0},
            {"sample_id": "S2", Split.COLUMN_NAME.value: Split.FIT.value, "label": 1},
        ]
    ).set_index("sample_id").to_parquet(metadata_path)

    dataset = MetadataBagsDataset(
        items_path=items_path,
        metadata_path=metadata_path,
        split=Split.FIT.value,
        task_kind="classification",
        target_key="metadata.label",
    )
    dataset.setup()
    batch = pad_bags_collate([dataset[0], dataset[1]])

    cfg = MILConfig(
        task=MILTaskConfig(kind="classification"),
        aggregator=AggregatorConfig(name="attention", hidden_dim=4, gated=True),
    )
    cfg.lit.target_key = "metadata.label"
    module = build_mil_module(cfg=cfg, input_dim=2, num_classes=2)
    module.log = lambda *args, **kwargs: None

    assert isinstance(module.aggregator, AttentionAggregation)
    output = module.training_step(batch, 0)
    assert torch.isfinite(output["loss"])
    assert output["prediction"].dtype == torch.long


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


def test_build_supervised_lit_returns_lazy_dataset(tmp_path: Path):
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

    lit, dataset = build_supervised_lit(cfg)

    assert lit.num_outputs == 1
    assert dataset.split is None
    assert dataset.id_key == "id"


def test_extract_mil_embeddings_uses_load_from_checkpoint_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = MILConfig()
    cfg.data.name = "beat"
    cfg.data.items_path = Path("items.json")
    cfg.data.metadata_path = Path("split.parquet")
    cfg.data.cache_dir = tmp_path / "cache"
    cfg.lit.target_key = "metadata.response"

    source_cfg = Config()
    source_cfg.task.target = "expression"
    source_cfg.lit.target_key = "target"
    source_cfg.data.name = "beat"
    source_cfg.data.items_path = Path("items.json")
    source_cfg.data.metadata_path = Path("metadata.parquet")
    source_cfg.data.panel_path = tmp_path / "panel.yaml"
    source_cfg.data.source_panel = ["A"]
    source_cfg.data.target_panel = ["B"]
    source_cfg.backbone.expr_encoder_name = "mlp"
    source_cfg.backbone.expr_encoder_kws = {"hidden_dim": 4, "output_dim": 4}
    source_cfg.data.panel_path.write_text("{}", encoding="utf-8")

    prediction_dir = cfg.data.cache_dir / "prediction"
    prediction_dir.mkdir(parents=True)
    torch.save(
        [{"id": ["tile-0"], "sample_id": ["S1"], "z": torch.tensor([[1.0, 2.0]])}],
        prediction_dir / "000000.pt",
    )

    calls: dict[str, Any] = {}

    def fake_prepare_training_config(current_cfg: Config):
        current_cfg.data.items_path = tmp_path / "supervised-items.json"
        current_cfg.data.metadata_path = tmp_path / "supervised-metadata.parquet"
        return SimpleNamespace(cfg=current_cfg, output_dir=tmp_path, num_source_genes=1, num_outputs=1)

    def fake_build_supervised_lit(current_cfg: Config):
        class DummyLit:
            backbone = object()
            head = object()
            num_outputs = 1
            batch_key = "modalities"
            target_key = "target"
            lr_head = 1e-4
            lr_backbone = 1e-5
            lr_alpha = 1e-3
            weight_decay = 1e-3
            eta = 1e-6
            schedule = "cosine"
            max_epochs = 35
            num_warmup_epochs = 5
            pooling = None

        class DummyDataset:
            def __init__(self, **kwargs):
                calls["dataset_kwargs"] = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def setup(self):
                calls["dataset_setup"] = True

        dataset = DummyDataset(
            target="expression",
            metadata_path=tmp_path / "supervised-metadata.parquet",
            source_panel=["A"],
            target_panel=["B"],
            include_image=False,
            include_expr=True,
            target_transform=None,
            image_transform=None,
            expr_transform=None,
            expr_pool="token",
            cache_dir=None,
            drop_nan_columns=True,
            id_key="id",
        )
        return DummyLit(), dataset

    class DummyLoadedLit:
        def eval(self):
            calls["eval"] = True

    def fake_load_from_checkpoint(checkpoint_path: Path, **kwargs):
        calls["checkpoint_path"] = checkpoint_path
        calls["load_kwargs"] = kwargs
        return DummyLoadedLit()

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def predict(self, *args, **kwargs):
            calls["predict_called"] = True

    monkeypatch.setattr("xenium_hne_fusion.train.mil.prepare_training_config", fake_prepare_training_config)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.build_supervised_lit", fake_build_supervised_lit)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.RegressionLit.load_from_checkpoint", fake_load_from_checkpoint)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.L.Trainer", DummyTrainer)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.DataLoader", lambda *args, **kwargs: None)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.write_mil_items_from_prediction_cache", lambda **kwargs: kwargs["items_path"])

    items_path = extract_mil_embeddings(
        resolved_run=SimpleNamespace(checkpoint_path=tmp_path / "best.ckpt", source_config=source_cfg),
        cfg=cfg,
        run_root=tmp_path / "run",
    )

    assert items_path == tmp_path / "run" / "mil-items.json"
    assert calls["checkpoint_path"] == tmp_path / "best.ckpt"
    assert calls["load_kwargs"]["target_key"] == "target"
    assert calls["dataset_kwargs"]["items_path"] == Path("items.json")
    assert calls["dataset_setup"] is True
    assert calls["eval"] is True
    assert "predict_called" not in calls


def test_extract_mil_embeddings_builds_minimal_prediction_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = MILConfig()
    cfg.data.name = "beat"
    cfg.data.items_path = Path("items.json")
    cfg.data.metadata_path = Path("split.parquet")
    cfg.data.cache_dir = tmp_path / "cache"

    source_cfg = Config()
    source_cfg.task.target = "expression"
    source_cfg.lit.target_key = "target"
    source_cfg.data.name = "beat"
    source_cfg.data.items_path = Path("items.json")
    source_cfg.data.metadata_path = Path("metadata.parquet")
    source_cfg.data.panel_path = tmp_path / "panel.yaml"
    source_cfg.data.source_panel = ["A"]
    source_cfg.data.target_panel = ["B"]
    source_cfg.backbone.expr_encoder_name = "mlp"
    source_cfg.backbone.expr_encoder_kws = {"hidden_dim": 4, "output_dim": 4}
    source_cfg.data.panel_path.write_text("{}", encoding="utf-8")

    calls: dict[str, Any] = {}

    def fake_prepare_training_config(current_cfg: Config):
        return SimpleNamespace(cfg=current_cfg, output_dir=tmp_path, num_source_genes=1, num_outputs=1)

    def fake_build_supervised_lit(current_cfg: Config):
        class DummyLit:
            backbone = object()
            head = object()
            num_outputs = 1
            batch_key = "modalities"
            target_key = "target"
            lr_head = 1e-4
            lr_backbone = 1e-5
            lr_alpha = 1e-3
            weight_decay = 1e-3
            eta = 1e-6
            schedule = "cosine"
            max_epochs = 35
            num_warmup_epochs = 5
            pooling = None

        class DummyDataset:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def setup(self):
                return None

        return DummyLit(), DummyDataset(
            target="expression",
            metadata_path=tmp_path / "supervised-metadata.parquet",
            source_panel=["A"],
            target_panel=["B"],
            include_image=False,
            include_expr=True,
            target_transform=None,
            image_transform=None,
            expr_transform=None,
            expr_pool="token",
            cache_dir=None,
            drop_nan_columns=True,
            id_key="id",
        )

    class DummyLoadedLit:
        def eval(self):
            return None

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def predict(self, *args, **kwargs):
            calls["predict_called"] = True

    monkeypatch.setattr("xenium_hne_fusion.train.mil.prepare_training_config", fake_prepare_training_config)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.build_supervised_lit", fake_build_supervised_lit)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.RegressionLit.load_from_checkpoint", lambda **kwargs: DummyLoadedLit())
    monkeypatch.setattr("xenium_hne_fusion.train.mil.L.Trainer", DummyTrainer)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.DataLoader", lambda *args, **kwargs: None)
    monkeypatch.setattr("xenium_hne_fusion.train.mil.write_mil_items_from_prediction_cache", lambda **kwargs: kwargs["items_path"])

    extract_mil_embeddings(
        resolved_run=SimpleNamespace(checkpoint_path=tmp_path / "best.ckpt", source_config=source_cfg),
        cfg=cfg,
        run_root=tmp_path / "run",
    )

    prediction_cache = calls["trainer_kwargs"]["callbacks"][0]
    assert prediction_cache.include_keys == ["z", "id", "sample_id"]
    assert calls["predict_called"] is True
