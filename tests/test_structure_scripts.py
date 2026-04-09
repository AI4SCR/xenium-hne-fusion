import importlib.util
from pathlib import Path

from xenium_hne_fusion.config import ProcessingConfig


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structure_hest1k_parser_uses_config_flag():
    module = _load_script("scripts/data/structure_hest1k.py", "structure_hest1k_script")
    processing_cfg = module.parse_processing_args(["--config", "configs/data/local/hest1k.yaml"], include_executor=False)[0]

    assert isinstance(processing_cfg, ProcessingConfig)
    assert processing_cfg.name == "hest1k"


def test_structure_beat_parser_uses_config_flag():
    module = _load_script("scripts/data/structure_beat.py", "structure_beat_script")
    processing_cfg = module.parse_processing_args(["--config", "configs/data/local/beat.yaml"], include_executor=False)[0]

    assert isinstance(processing_cfg, ProcessingConfig)
    assert processing_cfg.name == "beat"
