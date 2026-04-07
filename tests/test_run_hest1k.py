import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_hest1k_runs_full_pipeline_in_training_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    items_config_dir = tmp_path / "configs" / "items" / "hest1k"
    config_path.write_text(
        "name: hest1k\n"
        "tile_px: 256\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
        "filter:\n"
        "  sample_ids:\n"
        "    - NCBI783\n"
        "    - NCBI856\n"
    )
    items_config_dir.mkdir(parents=True, exist_ok=True)
    (items_config_dir / "default.yaml").write_text("name: default\norgans: null\nnum_transcripts: 100\n")
    (items_config_dir / "breast.yaml").write_text("name: breast\norgans:\n  - Breast\nnum_transcripts: 100\n")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_script")

    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    split_paths = {
        "default": tmp_path / "splits" / "default.yaml",
        "breast": tmp_path / "splits" / "breast.yaml",
    }
    calls = []

    def fake_get_hest_metadata_path(raw_dir_arg: Path) -> Path:
        calls.append(("get_hest_metadata_path", raw_dir_arg))
        return metadata_path

    def fake_create_structured_metadata_symlink(metadata_path_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(("create_structured_metadata_symlink", metadata_path_arg, structured_dir_arg))

    def fake_resolve_samples(cfg, metadata_path_arg: Path) -> list[str]:
        calls.append(("resolve_samples", cfg.filter.sample_ids, cfg.filter.species, cfg.filter.organ, metadata_path_arg))
        return ["B1", "L1"]

    def fake_filter_hest_samples_by_tile_mpp(cfg, sample_ids: list[str], metadata_path_arg: Path) -> list[str]:
        calls.append(("filter_hest_samples_by_tile_mpp", sample_ids, metadata_path_arg))
        return sample_ids

    def fake_ensure_hest_sample_downloaded(sample_id: str, raw_dir_arg: Path) -> None:
        calls.append(("ensure_hest_sample_downloaded", sample_id, raw_dir_arg))

    def fake_validate_hest_sample_mpp(sample_id: str, raw_dir_arg: Path, metadata_path_arg: Path) -> None:
        calls.append(("validate_hest_sample_mpp", sample_id, raw_dir_arg, metadata_path_arg))

    def fake_create_structured_symlinks(sample_id: str, raw_dir_arg: Path, structured_dir_arg: Path) -> None:
        calls.append(("create_structured_symlinks", sample_id, raw_dir_arg, structured_dir_arg))

    def fake_process_sample(cfg, sample_id: str, metadata_path_arg: Path, kernel_size: int, predicate: str, overwrite: bool) -> None:
        calls.append(("process_sample", sample_id, metadata_path_arg, kernel_size, predicate, overwrite))

    def fake_process_dataset_metadata(
        dataset: str,
        metadata_path: Path,
        output_path: Path,
        sample_ids: list[str] | None = None,
    ) -> None:
        calls.append(("process_dataset_metadata", dataset, metadata_path, output_path, sample_ids))

    def fake_create_all_items(cfg, kernel_size: int, overwrite: bool) -> Path:
        calls.append(("create_all_items", kernel_size, overwrite))
        return cfg.output_dir / "items" / "all.json"

    def fake_compute_all_tile_stats(cfg, cell_type_col: str, overwrite: bool) -> Path:
        calls.append(("compute_all_tile_stats", cell_type_col, overwrite))
        return cfg.output_dir / "statistics" / "all.parquet"

    def fake_create_filtered_items(cfg, items_config_path: Path, source_items_name: str, overwrite: bool) -> tuple[Path, int]:
        calls.append(("create_filtered_items", items_config_path.name, source_items_name, overwrite))
        name = module.load_items_filter_config(items_config_path).name
        return cfg.output_dir / "items" / f"{name}.json", 5

    def fake_get_split_config_path_for_items(items_config_path: Path) -> Path:
        name = module.load_items_filter_config(items_config_path).name
        calls.append(("get_split_config_path_for_items", name))
        return split_paths[name]

    def fake_create_split_collection(cfg, split_config_path: Path, items_path: Path, overwrite: bool) -> Path:
        calls.append(("create_split_collection", split_config_path, items_path, overwrite))
        return cfg.output_dir / "splits" / split_config_path.stem

    monkeypatch.setattr(module, "get_hest_metadata_path", fake_get_hest_metadata_path)
    monkeypatch.setattr(module, "create_structured_metadata_symlink", fake_create_structured_metadata_symlink)
    monkeypatch.setattr(module, "resolve_samples", fake_resolve_samples)
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", fake_filter_hest_samples_by_tile_mpp)
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", fake_ensure_hest_sample_downloaded)
    monkeypatch.setattr(module, "validate_hest_sample_mpp", fake_validate_hest_sample_mpp)
    monkeypatch.setattr(module, "create_structured_symlinks", fake_create_structured_symlinks)
    monkeypatch.setattr(module, "process_sample", fake_process_sample)
    monkeypatch.setattr(module, "process_dataset_metadata", fake_process_dataset_metadata)
    monkeypatch.setattr(module, "create_all_items", fake_create_all_items)
    monkeypatch.setattr(module, "compute_all_tile_stats", fake_compute_all_tile_stats)
    monkeypatch.setattr(module, "create_filtered_items", fake_create_filtered_items)
    monkeypatch.setattr(module, "get_split_config_path_for_items", fake_get_split_config_path_for_items)
    monkeypatch.setattr(module, "create_split_collection", fake_create_split_collection)

    module.main(
        config_path=config_path,
        items_config_dir=items_config_dir,
        kernel_size=32,
        predicate="intersects",
        cell_type_col="ct",
        overwrite=True,
    )

    processed_metadata_path = data_dir / "02_processed" / "hest1k" / "metadata.parquet"
    output_dir = data_dir / "03_output" / "hest1k"
    structured_dir = data_dir / "01_structured" / "hest1k"
    assert calls == [
        ("get_hest_metadata_path", raw_dir.resolve()),
        ("create_structured_metadata_symlink", metadata_path, structured_dir.resolve()),
        ("resolve_samples", None, "Homo sapiens", None, metadata_path),
        ("filter_hest_samples_by_tile_mpp", ["B1", "L1"], metadata_path),
        ("ensure_hest_sample_downloaded", "B1", raw_dir.resolve()),
        ("validate_hest_sample_mpp", "B1", raw_dir.resolve(), metadata_path),
        ("create_structured_symlinks", "B1", raw_dir.resolve(), structured_dir.resolve()),
        ("process_sample", "B1", metadata_path, 32, "intersects", True),
        ("ensure_hest_sample_downloaded", "L1", raw_dir.resolve()),
        ("validate_hest_sample_mpp", "L1", raw_dir.resolve(), metadata_path),
        ("create_structured_symlinks", "L1", raw_dir.resolve(), structured_dir.resolve()),
        ("process_sample", "L1", metadata_path, 32, "intersects", True),
        ("process_dataset_metadata", "hest1k", metadata_path, processed_metadata_path, ["B1", "L1"]),
        ("create_all_items", 32, True),
        ("compute_all_tile_stats", "ct", True),
        ("create_filtered_items", "default.yaml", "all", True),
        ("get_split_config_path_for_items", "default"),
        ("create_split_collection", split_paths["default"], output_dir / "items" / "default.json", True),
        ("create_filtered_items", "breast.yaml", "all", True),
        ("get_split_config_path_for_items", "breast"),
        ("create_split_collection", split_paths["breast"], output_dir / "items" / "breast.json", True),
    ]


def test_run_hest1k_skips_split_creation_for_empty_filtered_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    items_config_dir = tmp_path / "configs" / "items" / "hest1k"
    config_path.write_text(
        "name: hest1k\n"
        "tile_px: 256\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
        "filter:\n"
        "  sample_ids:\n"
        "    - NCBI783\n"
    )
    items_config_dir.mkdir(parents=True, exist_ok=True)
    (items_config_dir / "default.yaml").write_text("name: default\norgans: null\nnum_transcripts: 100\n")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_empty_items_script")
    calls = []
    metadata_path = raw_dir / "HEST_v1_3_0.csv"

    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: metadata_path)
    monkeypatch.setattr(module, "create_structured_metadata_symlink", lambda metadata_path_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", lambda cfg, sample_ids, metadata_path_arg: sample_ids)
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: None)
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: None)
    monkeypatch.setattr(module, "create_structured_symlinks", lambda sample_id, raw_dir_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "process_sample", lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: None)
    monkeypatch.setattr(module, "process_dataset_metadata", lambda dataset, metadata_path, output_path, sample_ids=None: None)
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: cfg.output_dir / "items" / "all.json")
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: cfg.output_dir / "statistics" / "all.parquet")
    monkeypatch.setattr(module, "create_filtered_items", lambda cfg, items_config_path, source_items_name, overwrite: (cfg.output_dir / "items" / "default.json", 0))
    monkeypatch.setattr(module, "create_split_collection", lambda cfg, split_config_path, items_path, overwrite: calls.append(("create_split_collection", split_config_path, items_path, overwrite)))
    monkeypatch.setattr(module, "resolve_samples", lambda cfg, metadata_path_arg: (_ for _ in ()).throw(AssertionError("resolve_samples should not be called")))

    module.main(
        config_path=config_path,
        items_config_dir=items_config_dir,
        sample_id="TENX116",
        overwrite=True,
    )

    assert calls == []


def test_run_hest1k_organ_flag_only_filters_final_items_and_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    items_config_dir = tmp_path / "configs" / "items" / "hest1k"
    config_path.write_text(
        "name: hest1k\n"
        "tile_px: 256\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
        "filter:\n"
        "  sample_ids:\n"
        "    - NCBI783\n"
    )
    items_config_dir.mkdir(parents=True, exist_ok=True)
    (items_config_dir / "default.yaml").write_text("name: default\norgans: null\nnum_transcripts: 100\n")
    (items_config_dir / "breast.yaml").write_text("name: breast\norgans:\n  - Breast\nnum_transcripts: 100\n")
    (items_config_dir / "lung.yaml").write_text("name: lung\norgans:\n  - Lung\nnum_transcripts: 100\n")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_organ_script")
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    calls = []

    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: metadata_path)
    monkeypatch.setattr(module, "create_structured_metadata_symlink", lambda metadata_path_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", lambda cfg, sample_ids, metadata_path_arg: sample_ids)
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: None)
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: None)
    monkeypatch.setattr(module, "create_structured_symlinks", lambda sample_id, raw_dir_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "process_sample", lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: None)
    monkeypatch.setattr(module, "process_dataset_metadata", lambda dataset, metadata_path, output_path, sample_ids=None: None)
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: cfg.output_dir / "items" / "all.json")
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: cfg.output_dir / "statistics" / "all.parquet")
    monkeypatch.setattr(module, "get_split_config_path_for_items", lambda items_config_path: tmp_path / "splits" / f"{module.load_items_filter_config(items_config_path).name}.yaml")
    monkeypatch.setattr(module, "create_split_collection", lambda cfg, split_config_path, items_path, overwrite: None)

    def fake_resolve_samples(cfg, metadata_path_arg: Path) -> list[str]:
        calls.append(("resolve_samples", cfg.filter.sample_ids, cfg.filter.species, cfg.filter.organ, metadata_path_arg))
        return ["B1", "L1"]

    def fake_create_filtered_items(cfg, items_config_path: Path, source_items_name: str, overwrite: bool) -> tuple[Path, int]:
        calls.append(("create_filtered_items", module.load_items_filter_config(items_config_path).name))
        name = module.load_items_filter_config(items_config_path).name
        return cfg.output_dir / "items" / f"{name}.json", 5

    monkeypatch.setattr(module, "resolve_samples", fake_resolve_samples)
    monkeypatch.setattr(module, "create_filtered_items", fake_create_filtered_items)

    module.main(
        config_path=config_path,
        items_config_dir=items_config_dir,
        organ="Breast",
        overwrite=True,
    )

    assert calls == [
        ("resolve_samples", None, "Homo sapiens", None, metadata_path),
        ("create_filtered_items", "default"),
        ("create_filtered_items", "breast"),
    ]


def test_filter_hest_samples_by_tile_mpp_keeps_only_eligible_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_filter_script")
    cfg = module.PipelineConfig(
        dataset="hest1k",
        name="hest1k",
        tile_px=256,
        stride_px=256,
        tile_mpp=0.5,
        raw_dir=tmp_path / "raw",
        structured_dir=tmp_path / "structured",
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )
    metadata_path = tmp_path / "HEST_v1_3_0.csv"
    slide_mpps = {"A": 0.5, "B": 0.25, "C": 1.0}

    monkeypatch.setattr(module, "get_hest_sample_mpp", lambda sample_id, _: slide_mpps[sample_id])

    assert module.filter_hest_samples_by_tile_mpp(cfg, ["A", "B", "C"], metadata_path) == ["A", "B"]


def test_filter_hest_samples_by_tile_mpp_raises_for_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_filter_error_script")
    cfg = module.PipelineConfig(
        dataset="hest1k",
        name="hest1k",
        tile_px=256,
        stride_px=256,
        tile_mpp=0.5,
        raw_dir=tmp_path / "raw",
        structured_dir=tmp_path / "structured",
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )
    metadata_path = tmp_path / "HEST_v1_3_0.csv"

    def fake_get_hest_sample_mpp(sample_id: str, _: Path) -> float:
        raise AssertionError(f"pixel_size_um_embedded missing for {sample_id}")

    monkeypatch.setattr(module, "get_hest_sample_mpp", fake_get_hest_sample_mpp)

    with pytest.raises(AssertionError, match="pixel_size_um_embedded missing for A"):
        module.filter_hest_samples_by_tile_mpp(cfg, ["A"], metadata_path)


def test_is_hest_sample_processed_requires_exact_tile_match(
    tmp_path: Path,
):
    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_processed_script")
    cfg = module.PipelineConfig(
        dataset="hest1k",
        name="hest1k",
        tile_px=256,
        stride_px=256,
        tile_mpp=0.5,
        raw_dir=tmp_path / "raw",
        structured_dir=tmp_path / "structured",
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )
    sample_id = "TENX116"
    tiles_path = cfg.structured_dir / sample_id / "tiles" / "256_256.parquet"
    tiles_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tile_id": [0, 1]}).to_parquet(tiles_path)

    processed_dir = cfg.processed_dir / sample_id / "256_256"
    for tile_id in ["0", "1"]:
        tile_dir = processed_dir / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)
        (tile_dir / "tile.pt").write_text("")
        (tile_dir / "transcripts.parquet").write_text("")
        (tile_dir / "expr-kernel_size=16.parquet").write_text("")

    assert module.is_hest_sample_processed(cfg, sample_id, kernel_size=16)

    extra_tile_dir = processed_dir / "2"
    extra_tile_dir.mkdir()
    (extra_tile_dir / "tile.pt").write_text("")
    (extra_tile_dir / "transcripts.parquet").write_text("")
    (extra_tile_dir / "expr-kernel_size=16.parquet").write_text("")

    assert not module.is_hest_sample_processed(cfg, sample_id, kernel_size=16)


def test_run_hest1k_skips_processing_for_completed_samples_and_keeps_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    items_config_dir = tmp_path / "configs" / "items" / "hest1k"
    config_path.write_text(
        "name: hest1k\n"
        "tile_px: 256\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
    )
    items_config_dir.mkdir(parents=True, exist_ok=True)
    (items_config_dir / "default.yaml").write_text("name: default\norgans: null\nnum_transcripts: 100\n")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_resume_script")
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    calls = []

    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: metadata_path)
    monkeypatch.setattr(module, "create_structured_metadata_symlink", lambda metadata_path_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "resolve_samples", lambda cfg, metadata_path_arg: ["B1", "L1"])
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", lambda cfg, sample_ids, metadata_path_arg: sample_ids)
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: calls.append(("ensure", sample_id)))
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: calls.append(("validate", sample_id)))
    monkeypatch.setattr(module, "create_structured_symlinks", lambda sample_id, raw_dir_arg, structured_dir_arg: calls.append(("symlink", sample_id)))
    monkeypatch.setattr(module, "process_sample", lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: calls.append(("process", sample_id)))
    monkeypatch.setattr(module, "is_hest_sample_processed", lambda cfg, sample_id, kernel_size: sample_id == "B1")
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: cfg.output_dir / "items" / "all.json")
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: cfg.output_dir / "statistics" / "all.parquet")
    monkeypatch.setattr(module, "create_filtered_items", lambda cfg, items_config_path, source_items_name, overwrite: (cfg.output_dir / "items" / "default.json", 0))

    module.main(
        config_path=config_path,
        items_config_dir=items_config_dir,
        overwrite=False,
    )

    assert calls == [
        ("ensure", "L1"),
        ("validate", "L1"),
        ("symlink", "L1"),
        ("process", "L1"),
        ("metadata", ["B1", "L1"]),
    ]


def test_run_hest1k_excludes_ineligible_samples_before_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    items_config_dir = tmp_path / "configs" / "items" / "hest1k"
    config_path.write_text(
        "name: hest1k\n"
        "tile_px: 256\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
    )
    items_config_dir.mkdir(parents=True, exist_ok=True)
    (items_config_dir / "default.yaml").write_text("name: default\norgans: null\nnum_transcripts: 100\n")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_eligible_script")
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    calls = []

    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: metadata_path)
    monkeypatch.setattr(module, "create_structured_metadata_symlink", lambda metadata_path_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "resolve_samples", lambda cfg, metadata_path_arg: ["B1", "L1"])
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", lambda cfg, sample_ids, metadata_path_arg: ["L1"])
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: calls.append(("ensure", sample_id)))
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: calls.append(("validate", sample_id)))
    monkeypatch.setattr(module, "create_structured_symlinks", lambda sample_id, raw_dir_arg, structured_dir_arg: calls.append(("symlink", sample_id)))
    monkeypatch.setattr(module, "process_sample", lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: calls.append(("process", sample_id)))
    monkeypatch.setattr(module, "is_hest_sample_processed", lambda cfg, sample_id, kernel_size: False)
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: cfg.output_dir / "items" / "all.json")
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: cfg.output_dir / "statistics" / "all.parquet")
    monkeypatch.setattr(module, "create_filtered_items", lambda cfg, items_config_path, source_items_name, overwrite: (cfg.output_dir / "items" / "default.json", 0))

    module.main(
        config_path=config_path,
        items_config_dir=items_config_dir,
        overwrite=False,
    )

    assert calls == [
        ("ensure", "L1"),
        ("validate", "L1"),
        ("symlink", "L1"),
        ("process", "L1"),
        ("metadata", ["L1"]),
    ]
