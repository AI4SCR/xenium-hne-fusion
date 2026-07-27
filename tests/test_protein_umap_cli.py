import importlib.util
from pathlib import Path

from xenium_hne_fusion.targets import PROTEIN_PANEL


def _load_protein_umap_module():
    path = Path("scripts/eval/protein_umap.py").resolve()
    spec = importlib.util.spec_from_file_location("protein_umap_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_protein_umap_cli_builds_config():
    module = _load_protein_umap_module()
    parser = module._build_parser()
    cfg = parser.parse_args(["--umap.name", "owkin", "--umap.data_dir", "/tmp/data"])
    init = parser.instantiate(cfg)

    assert init.umap.name == "owkin"
    assert init.umap.data_dir == Path("/tmp/data")
    assert init.umap.items_path == Path("all.json")
    assert init.umap.proteins == PROTEIN_PANEL
    assert init.overwrite is False


def test_protein_umap_cli_overwrite_flag():
    module = _load_protein_umap_module()
    parser = module._build_parser()
    cfg = parser.parse_args(
        ["--umap.name", "owkin", "--umap.data_dir", "/tmp/data", "--overwrite", "true"]
    )
    init = parser.instantiate(cfg)
    assert init.overwrite is True
