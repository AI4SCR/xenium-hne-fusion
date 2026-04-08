import importlib.util
from pathlib import Path

import pytest

from xenium_hne_fusion.utils.getters import ManagedPaths, PipelineConfig, ProcessingConfig, TilesConfig, load_processing_config


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeFuture:
    def __init__(self, fake_ray, fn, name: str, args: tuple[object, ...], sample_id: str):
        self.fake_ray = fake_ray
        self.fn = fn
        self.name = name
        self.args = args
        self.sample_id = sample_id
        self._resolved = False
        self._result = None

    def resolve(self):
        if self._resolved:
            return self._result
        if self.args and isinstance(self.args[0], _FakeFuture):
            self.args[0].resolve()
        if self.name == "process_sample_remote" and self.sample_id in self.fake_ray.fail_samples:
            raise RuntimeError(f"boom {self.sample_id}")
        self._result = self.fn(*self.args)
        self.fake_ray.calls.append(("resolve", self.name, self.sample_id))
        self._resolved = True
        return self._result


class _FakeRemoteFunction:
    def __init__(self, fake_ray, fn, name: str):
        self.fake_ray = fake_ray
        self.fn = fn
        self.name = name

    def remote(self, *args):
        if self.name == "structure_sample_remote":
            sample_id = args[1]
        elif self.name == "detect_tissues_remote":
            sample_id = args[2]
        else:
            sample_id = args[2]
        depends_on_future = bool(args) and isinstance(args[0], _FakeFuture)
        self.fake_ray.calls.append(("submit", self.name, sample_id, depends_on_future))
        return _FakeFuture(self.fake_ray, self.fn, self.name, args, sample_id)


class _FakeRay:
    def __init__(self, calls: list[tuple], fail_samples: set[str] | None = None):
        self.calls = calls
        self.fail_samples = fail_samples or set()
        self.initialized = False

    def is_initialized(self) -> bool:
        self.calls.append(("ray.is_initialized",))
        return self.initialized

    def init(self) -> None:
        self.calls.append(("ray.init",))
        self.initialized = True

    def remote(self, **_resources):
        def decorator(fn):
            return _FakeRemoteFunction(self, fn, fn.__name__)

        return decorator

    def get(self, future: _FakeFuture):
        self.calls.append(("ray.get", future.name, future.sample_id))
        return future.resolve()


def test_run_hest1k_runs_full_pipeline_with_unified_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    config_path.write_text(
        "name: hest1k\n"
        "tiles:\n"
        "  tile_px: 256\n"
        "  stride_px: 256\n"
        "  mpp: 0.5\n"
        "  img_size: 224\n"
        "  kernel_size: 16\n"
        "  predicate: within\n"
        "filter:\n"
        "  species: Homo sapiens\n"
        "  include_ids:\n"
        "    - NCBI783\n"
        "    - NCBI856\n"
        "  exclude_ids: null\n"
        "items:\n"
        "  name: default\n"
        "  filter:\n"
        "    organs: null\n"
        "    num_transcripts: 100\n"
        "split:\n"
        "  name: default\n"
        "  test_size: 0.25\n"
        "  val_size: 0.25\n"
        "  random_state: 0\n"
    )
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_script")
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    calls = []

    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: metadata_path)
    monkeypatch.setattr(
        module,
        "create_structured_metadata_symlink",
        lambda metadata_path_arg, structured_dir_arg: calls.append(("structured_metadata", metadata_path_arg, structured_dir_arg)),
    )
    monkeypatch.setattr(
        module,
        "resolve_samples",
        lambda cfg, metadata_path_arg: (
            calls.append(("resolve_samples", cfg.filter.include_ids, cfg.filter.exclude_ids, cfg.filter.species, cfg.filter.organ, metadata_path_arg)),
            ["B1", "L1"],
        )[1],
    )
    monkeypatch.setattr(
        module,
        "filter_hest_samples_by_tile_mpp",
        lambda cfg, sample_ids, metadata_path_arg: (
            calls.append(("filter_hest_samples_by_tile_mpp", sample_ids, metadata_path_arg)),
            sample_ids,
        )[1],
    )
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: calls.append(("ensure", sample_id)))
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: calls.append(("validate", sample_id)))
    monkeypatch.setattr(
        module,
        "create_structured_symlinks",
        lambda sample_id, raw_dir_arg, structured_dir_arg: (
            (structured_dir_arg / sample_id).mkdir(parents=True, exist_ok=True),
            calls.append(("symlink", sample_id)),
        ),
    )
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: calls.append(("detect", sample_id)))
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True),
            calls.append(("process", sample_id, kernel_size, predicate, overwrite)),
        ),
    )
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path=None, output_path=None, selected_sample_ids=None: calls.append(
            ("metadata", dataset, metadata_path, output_path, selected_sample_ids)
        ),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: calls.append(("items", kernel_size, overwrite)))
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: calls.append(("stats", cell_type_col, overwrite)))
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, overwrite=False: (
            calls.append(("filtered", cfg.items.name, overwrite)),
            (cfg.output_dir / "items" / "default.json", 5),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "create_split_collection",
        lambda cfg, items_path, overwrite=False: calls.append(
            ("split", cfg.split.name, items_path, overwrite)
        ),
    )

    processing_cfg = load_processing_config(config_path)
    processing_cfg.tiles.kernel_size = 32
    processing_cfg.tiles.predicate = "intersects"
    module.main(processing_cfg=processing_cfg, cell_type_col="ct", overwrite=True)

    processed_metadata_path = data_dir / "02_processed" / "hest1k" / "metadata.parquet"
    output_dir = data_dir / "03_output" / "hest1k"
    structured_dir = data_dir / "01_structured" / "hest1k"
    assert calls == [
        ("structured_metadata", metadata_path, structured_dir.resolve()),
        ("resolve_samples", ["NCBI783", "NCBI856"], None, "Homo sapiens", None, metadata_path),
        ("filter_hest_samples_by_tile_mpp", ["B1", "L1"], metadata_path),
        ("ensure", "B1"),
        ("validate", "B1"),
        ("symlink", "B1"),
        ("detect", "B1"),
        ("process", "B1", 32, "intersects", True),
        ("ensure", "L1"),
        ("validate", "L1"),
        ("symlink", "L1"),
        ("detect", "L1"),
        ("process", "L1", 32, "intersects", True),
        ("metadata", "hest1k", metadata_path, processed_metadata_path, ["B1", "L1"]),
        ("items", 32, True),
        ("stats", "ct", True),
        ("filtered", "default", True),
        ("split", "default", output_dir / "items" / "default.json", True),
    ]


def test_run_hest1k_local_and_remote_configs_define_sample_scope():
    from xenium_hne_fusion.utils.getters import load_dataset_config

    local_cfg = load_dataset_config(Path("configs/data/local/hest1k.yaml"))
    remote_cfg = load_dataset_config(Path("configs/data/remote/hest1k.yaml"))

    assert local_cfg.filter.include_ids == ["NCBI783", "NCBI856", "TENX116"]
    assert local_cfg.filter.exclude_ids is None
    assert local_cfg.items.filter.include_ids is None
    assert local_cfg.items.filter.exclude_ids is None
    assert remote_cfg.filter.include_ids is None
    assert remote_cfg.filter.exclude_ids is None
    assert remote_cfg.items.filter.include_ids is None
    assert remote_cfg.items.filter.exclude_ids is None


def test_filter_hest_samples_by_tile_mpp_keeps_only_eligible_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_filter_script")
    cfg = PipelineConfig(
        dataset="hest1k",
        raw_dir=tmp_path / "raw",
        paths=ManagedPaths(
            data_dir=tmp_path / "data",
            structured_dir=tmp_path / "structured",
            processed_dir=tmp_path / "processed",
            output_dir=tmp_path / "output",
        ),
        processing=ProcessingConfig(name="hest1k", tiles=TilesConfig(tile_px=256, stride_px=256, mpp=0.5, img_size=224)),
    )
    metadata_path = tmp_path / "HEST_v1_3_0.csv"
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: sample_id != "C")

    assert module.filter_hest_samples_by_tile_mpp(cfg, ["A", "B", "C"], metadata_path) == ["A", "B"]


def test_sample_done_markers_define_completion(tmp_path: Path):
    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_processed_script")
    cfg = PipelineConfig(
        dataset="hest1k",
        raw_dir=tmp_path / "raw",
        paths=ManagedPaths(
            data_dir=tmp_path / "data",
            structured_dir=tmp_path / "structured",
            processed_dir=tmp_path / "processed",
            output_dir=tmp_path / "output",
        ),
        processing=ProcessingConfig(name="hest1k", tiles=TilesConfig(tile_px=256, stride_px=256, mpp=0.5, img_size=224)),
    )

    (cfg.structured_dir / "S1").mkdir(parents=True, exist_ok=True)
    (cfg.processed_dir / "S1" / "256_256").mkdir(parents=True, exist_ok=True)

    assert not module.is_sample_structured(cfg, "S1")
    assert not module.is_sample_processed(cfg, "S1")

    module.mark_sample_structured(cfg, "S1")
    module.mark_sample_processed(cfg, "S1")

    assert module.is_sample_structured(cfg, "S1")
    assert module.is_sample_processed(cfg, "S1")


def test_run_hest1k_ray_chains_samples_and_finalizes_after_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "hest1k"
    config_path = tmp_path / "hest1k.yaml"
    config_path.write_text(
        "name: hest1k\n"
        "tiles:\n"
        "  tile_px: 256\n"
        "  stride_px: 256\n"
        "  mpp: 0.5\n"
        "  img_size: 224\n"
        "filter:\n"
        "  species: Homo sapiens\n"
        "  include_ids: null\n"
        "  exclude_ids: null\n"
        "items:\n"
        "  name: default\n"
        "  filter:\n"
        "    organs: null\n"
        "split:\n"
        "  name: default\n"
        "  test_size: 0.25\n"
        "  val_size: 0.25\n"
        "  random_state: 0\n"
    )
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEST1K_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_hest1k.py", "run_hest1k_ray_script")
    calls = []
    fake_ray = _FakeRay(calls)

    monkeypatch.setattr(module, "load_ray_module", lambda: fake_ray)
    monkeypatch.setattr(module, "get_hest_metadata_path", lambda raw_dir_arg: raw_dir / "HEST_v1_3_0.csv")
    monkeypatch.setattr(module, "create_structured_metadata_symlink", lambda metadata_path_arg, structured_dir_arg: None)
    monkeypatch.setattr(module, "resolve_samples", lambda cfg, metadata_path_arg: ["DONE", "L1", "P1"])
    monkeypatch.setattr(module, "filter_hest_samples_by_tile_mpp", lambda cfg, sample_ids, metadata_path_arg: sample_ids)
    monkeypatch.setattr(module, "can_extract_sample_at_tile_mpp", lambda cfg, sample_id, metadata_path_arg: True)
    monkeypatch.setattr(module, "ensure_hest_sample_downloaded", lambda sample_id, raw_dir_arg: calls.append(("ensure", sample_id)))
    monkeypatch.setattr(module, "validate_hest_sample_mpp", lambda sample_id, raw_dir_arg, metadata_path_arg: calls.append(("validate", sample_id)))
    monkeypatch.setattr(
        module,
        "create_structured_symlinks",
        lambda sample_id, raw_dir_arg, structured_dir_arg: (
            (structured_dir_arg / sample_id).mkdir(parents=True, exist_ok=True),
            calls.append(("symlink", sample_id)),
        ),
    )
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: calls.append(("detect", sample_id)))
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, metadata_path_arg, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True),
            calls.append(("process", sample_id, kernel_size, predicate, overwrite)),
        ),
    )
    monkeypatch.setattr(module, "process_dataset_metadata", lambda dataset, metadata_path, output_path, selected_sample_ids=None: calls.append(("metadata", selected_sample_ids)))
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: calls.append(("items", kernel_size, overwrite)))
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: calls.append(("stats", cell_type_col, overwrite)))
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, overwrite=False: (
            calls.append(("filtered", cfg.items.name, overwrite)),
            (cfg.output_dir / "items" / "default.json", 1),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "create_split_collection",
        lambda cfg, items_path, overwrite=False: calls.append(("split", cfg.split.name, items_path, overwrite)),
    )

    cfg = module.load_pipeline_config("hest1k", config_path)
    (cfg.structured_dir / "DONE").mkdir(parents=True, exist_ok=True)
    (cfg.processed_dir / "DONE" / "256_256").mkdir(parents=True, exist_ok=True)
    module.mark_sample_structured(cfg, "DONE")
    module.mark_sample_processed(cfg, "DONE")
    (cfg.structured_dir / "P1").mkdir(parents=True, exist_ok=True)
    module.mark_sample_structured(cfg, "P1")

    processing_cfg = load_processing_config(config_path)
    module.main(processing_cfg=processing_cfg, overwrite=False, executor="ray")

    assert ("metadata", ["DONE", "L1", "P1"]) in calls
    assert ("items", 16, False) in calls
    assert ("stats", "Level3_grouped", False) in calls
    assert ("filtered", "default", False) in calls
    assert ("split", "default", cfg.output_dir / "items" / "default.json", False) in calls
