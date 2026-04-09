import importlib.util
from pathlib import Path

from xenium_hne_fusion.train.config import (
    BackboneConfig,
    Config,
    DataLoaderConfig,
    HeadConfig,
    LitConfig,
    TaskConfig,
    TrainerConfig,
    WandbConfig,
)


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_supervised_parser_reads_yaml_config_into_namespace():
    module = _load_script("scripts/train/supervised.py", "train_supervised_script")
    parser = module._build_parser()

    namespace = parser.parse_args(["--config", "configs/train/beat/expression/default/early-fusion.yaml"])
    data = namespace.as_dict()

    assert data["debug"] is False
    assert data["task"]["target"] == "expression"
    assert data["backbone"]["morph_encoder_name"] == "vit_small_patch16_224"
    assert data["data"]["items_path"] == Path("default.json")
    assert data["data"]["metadata_path"] == Path("default/outer=0-inner=0-seed=0.parquet")
    assert data["data"]["panel_path"] == Path("default.yaml")


def test_supervised_namespace_bridge_returns_concrete_training_config():
    module = _load_script("scripts/train/supervised.py", "train_supervised_bridge_script")
    parser = module._build_parser()
    namespace = parser.parse_args(["--config", "configs/train/beat/expression/default/early-fusion.yaml"])

    cfg = module._namespace_to_config(namespace)

    assert isinstance(cfg, Config)
    assert isinstance(cfg.task, TaskConfig)
    assert isinstance(cfg.head, HeadConfig)
    assert isinstance(cfg.backbone, BackboneConfig)
    assert isinstance(cfg.data, DataLoaderConfig)
    assert isinstance(cfg.lit, LitConfig)
    assert isinstance(cfg.trainer, TrainerConfig)
    assert isinstance(cfg.wandb, WandbConfig)
    assert cfg.task.target == "expression"
    assert cfg.data.items_path == Path("default.json")
    assert cfg.data.metadata_path == Path("default/outer=0-inner=0-seed=0.parquet")
    assert cfg.data.panel_path == Path("default.yaml")
