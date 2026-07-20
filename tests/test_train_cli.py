import importlib.util
from pathlib import Path

from xenium_hne_fusion.train.config import TrainingConfig

CONFIG_PATH = "configs/train/owkin/proteins/early-fusion.yaml"


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_supervised_parser_reads_yaml_config_into_nested_namespace():
    module = _load_script("scripts/train/supervised.py", "train_supervised_script")
    parser = module._build_parser()

    namespace = parser.parse_args(["--config", CONFIG_PATH])
    data = namespace.as_dict()

    assert data["debug"] is False
    assert data["train"]["task"]["target"] == "proteins"
    assert data["train"]["backbone"]["morph_encoder_name"] == "vit_small_patch16_224"
    assert data["train"]["data"]["items_path"] == Path("c_cells.json")
    assert data["train"]["data"]["metadata_path"] == Path("c_cells/outer=0.parquet")
    assert data["train"]["data"]["panel_path"] == Path("c_cells.yaml")


def test_supervised_cli_instantiates_concrete_training_config_and_calls_main(monkeypatch):
    module = _load_script("scripts/train/supervised.py", "train_supervised_cli_script")

    captured = {}

    def fake_main(cfg, debug=None, config_path=None):
        captured["cfg"] = cfg
        captured["debug"] = debug
        captured["config_path"] = config_path

    monkeypatch.setattr(module, "main", fake_main)

    exit_code = module.cli(["--config", CONFIG_PATH])

    assert exit_code == 0
    cfg = captured["cfg"]
    assert isinstance(cfg, TrainingConfig)
    assert cfg.task.target == "proteins"
    assert cfg.data.items_path == Path("c_cells.json")
    assert cfg.data.metadata_path == Path("c_cells/outer=0.parquet")
    assert cfg.data.panel_path == Path("c_cells.yaml")
    assert captured["debug"] is False
