import importlib.util
from pathlib import Path

from xenium_hne_fusion.targets import PROTEIN_PANEL


def _load_batch_correction_module():
    path = Path("scripts/eval/batch_correction.py").resolve()
    spec = importlib.util.spec_from_file_location("batch_correction_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_batch_correction_cli_builds_config():
    module = _load_batch_correction_module()
    parser = module._build_parser()
    cfg = parser.parse_args(["--batch_correction.name", "owkin", "--batch_correction.data_dir", "/tmp/data"])
    init = parser.instantiate(cfg)

    assert init.batch_correction.name == "owkin"
    assert init.batch_correction.data_dir == Path("/tmp/data")
    assert init.batch_correction.items_path == Path("c_cells.json")
    assert init.batch_correction.proteins == PROTEIN_PANEL
    assert init.batch_correction.debug is False


def test_batch_correction_cli_debug_flag():
    module = _load_batch_correction_module()
    parser = module._build_parser()
    cfg = parser.parse_args([
        "--batch_correction.name", "owkin",
        "--batch_correction.data_dir", "/tmp/data",
        "--batch_correction.debug", "true",
        "--batch_correction.debug_cells_per_batch", "100",
    ])
    init = parser.instantiate(cfg)

    assert init.batch_correction.debug is True
    assert init.batch_correction.debug_cells_per_batch == 100
