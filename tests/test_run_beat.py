import importlib.util
from pathlib import Path

import pytest

from xenium_hne_fusion.utils.getters import load_processing_config


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


def test_run_beat_runs_full_pipeline_in_training_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "beat"
    config_path = tmp_path / "beat.yaml"
    config_path.write_text(
        "name: beat\n"
        "tile_px: 512\n"
        "stride_px: 256\n"
        "tile_mpp: 0.5\n"
        "filter:\n"
        "  sample_ids:\n"
        "    - S1\n"
        "    - S2\n"
    )
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("BEAT_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_beat.py", "run_beat_script")
    calls = []

    monkeypatch.setattr(module, "structure_beat_metadata", lambda cfg: calls.append(("structure_metadata", cfg.raw_dir)))
    monkeypatch.setattr(module, "resolve_beat_samples", lambda cfg, sample_id=None: ["S1", "S2"])
    monkeypatch.setattr(module, "structure_sample_stage", lambda cfg, sample_id: calls.append(("structure", sample_id)))
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: calls.append(("detect", sample_id)))
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True),
            calls.append(("process", sample_id, kernel_size, predicate, overwrite)),
        ),
    )
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: calls.append(("items", kernel_size, overwrite)))
    monkeypatch.setattr(
        module,
        "compute_all_tile_stats",
        lambda cfg, cell_type_col, overwrite: calls.append(("stats", cell_type_col, overwrite)),
    )
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, source_items_name="all", overwrite=False: (
            calls.append(("filtered", cfg.items.name, source_items_name, overwrite)),
            (cfg.output_dir / "items" / "default.json", 5),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "create_split_collection",
        lambda cfg, items_path, overwrite=False: calls.append(("split", cfg.split.split_name, items_path, overwrite)),
    )

    processing_cfg = load_processing_config(config_path)
    processing_cfg.tiles.kernel_size = 32
    processing_cfg.tiles.predicate = "intersects"
    module.main(processing_cfg=processing_cfg, cell_type_col="ct", overwrite=True)

    output_dir = data_dir / "03_output" / "beat"
    assert calls == [
        ("structure_metadata", raw_dir.resolve()),
        ("structure", "S1"),
        ("detect", "S1"),
        ("process", "S1", 32, "intersects", True),
        ("structure", "S2"),
        ("detect", "S2"),
        ("process", "S2", 32, "intersects", True),
        ("metadata", ["S1", "S2"]),
        ("items", 32, True),
        ("stats", "ct", True),
        ("filtered", "default", "all", True),
        ("split", "default", output_dir / "items" / "default.json", True),
    ]


def test_run_beat_skips_processing_for_completed_samples_and_keeps_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "beat"
    config_path = tmp_path / "beat.yaml"
    config_path.write_text("name: beat\ntile_px: 512\nstride_px: 256\ntile_mpp: 0.5\n")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("BEAT_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_beat.py", "run_beat_resume_script")
    calls = []

    monkeypatch.setattr(module, "structure_beat_metadata", lambda cfg: None)
    monkeypatch.setattr(module, "resolve_beat_samples", lambda cfg, sample_id=None: ["DONE", "S2"])
    monkeypatch.setattr(module, "structure_sample_stage", lambda cfg, sample_id: calls.append(("structure", sample_id)))
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: calls.append(("detect", sample_id)))
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True),
            calls.append(("process", sample_id)),
        ),
    )
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: None)
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: None)
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, source_items_name="all", overwrite=False: (cfg.output_dir / "items" / "default.json", 0),
    )

    cfg = module.load_pipeline_config("beat", config_path)
    (cfg.structured_dir / "DONE").mkdir(parents=True, exist_ok=True)
    (cfg.processed_dir / "DONE" / "512_256").mkdir(parents=True, exist_ok=True)
    module.mark_sample_structured(cfg, "DONE")
    module.mark_sample_processed(cfg, "DONE")

    processing_cfg = load_processing_config(config_path)
    module.main(processing_cfg=processing_cfg, overwrite=False)

    assert calls == [
        ("structure", "S2"),
        ("detect", "S2"),
        ("process", "S2"),
        ("metadata", ["DONE", "S2"]),
    ]


def test_run_beat_ray_chains_samples_and_finalizes_after_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "beat"
    config_path = tmp_path / "beat.yaml"
    config_path.write_text("name: beat\ntile_px: 512\nstride_px: 256\ntile_mpp: 0.5\n")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("BEAT_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_beat.py", "run_beat_ray_script")
    calls = []
    fake_ray = _FakeRay(calls)

    monkeypatch.setattr(module, "load_ray_module", lambda: fake_ray)
    monkeypatch.setattr(module, "structure_beat_metadata", lambda cfg: None)
    monkeypatch.setattr(module, "resolve_beat_samples", lambda cfg, sample_id=None: ["DONE", "S1", "S2"])
    monkeypatch.setattr(module, "structure_sample_stage", lambda cfg, sample_id: calls.append(("structure", sample_id)))
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: calls.append(("detect", sample_id)))
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True),
            calls.append(("process", sample_id, kernel_size, predicate, overwrite)),
        ),
    )
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: calls.append(("items", kernel_size, overwrite)))
    monkeypatch.setattr(
        module,
        "compute_all_tile_stats",
        lambda cfg, cell_type_col, overwrite: calls.append(("stats", cell_type_col, overwrite)),
    )
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, source_items_name="all", overwrite=False: (
            calls.append(("filtered", cfg.items.name, source_items_name, overwrite)),
            (cfg.output_dir / "items" / "default.json", 0),
        )[1],
    )

    cfg = module.load_pipeline_config("beat", config_path)
    (cfg.structured_dir / "DONE").mkdir(parents=True, exist_ok=True)
    (cfg.processed_dir / "DONE" / "512_256").mkdir(parents=True, exist_ok=True)
    module.mark_sample_structured(cfg, "DONE")
    module.mark_sample_processed(cfg, "DONE")
    (cfg.structured_dir / "S2").mkdir(parents=True, exist_ok=True)
    module.mark_sample_structured(cfg, "S2")

    processing_cfg = load_processing_config(config_path)
    processing_cfg.tiles.kernel_size = 32
    processing_cfg.tiles.predicate = "intersects"
    module.main(processing_cfg=processing_cfg, cell_type_col="ct", overwrite=False, executor="ray")

    assert ("submit", "structure_sample_remote", "S1", False) in calls
    assert ("submit", "detect_tissues_remote", "S1", True) in calls
    assert ("submit", "process_sample_remote", "S1", True) in calls
    assert ("submit", "detect_tissues_remote", "S2", False) in calls
    assert ("submit", "process_sample_remote", "S2", True) in calls
    assert ("detect", "S1") in calls
    assert ("detect", "S2") in calls
    assert ("process", "S1", 32, "intersects", False) in calls
    assert ("process", "S2", 32, "intersects", False) in calls
    assert ("metadata", ["DONE", "S1", "S2"]) in calls
    assert calls.index(("metadata", ["DONE", "S1", "S2"])) > calls.index(("resolve", "process_sample_remote", "S1"))
    assert calls.index(("metadata", ["DONE", "S1", "S2"])) > calls.index(("resolve", "process_sample_remote", "S2"))


def test_run_beat_ray_aborts_finalization_when_any_sample_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw" / "beat"
    config_path = tmp_path / "beat.yaml"
    config_path.write_text("name: beat\ntile_px: 512\nstride_px: 256\ntile_mpp: 0.5\n")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("BEAT_RAW_DIR", str(raw_dir))

    module = _load_script("scripts/data/run_beat.py", "run_beat_ray_failure_script")
    calls = []
    fake_ray = _FakeRay(calls, fail_samples={"S1"})

    monkeypatch.setattr(module, "load_ray_module", lambda: fake_ray)
    monkeypatch.setattr(module, "structure_beat_metadata", lambda cfg: None)
    monkeypatch.setattr(module, "resolve_beat_samples", lambda cfg, sample_id=None: ["S1", "S2"])
    monkeypatch.setattr(module, "structure_sample_stage", lambda cfg, sample_id: None)
    monkeypatch.setattr(module, "detect_sample_tissues", lambda cfg, sample_id: None)
    monkeypatch.setattr(
        module,
        "process_sample",
        lambda cfg, sample_id, kernel_size, predicate, overwrite: (
            (cfg.processed_dir / sample_id / f"{cfg.tile_px}_{cfg.stride_px}").mkdir(parents=True, exist_ok=True)
        ),
    )
    monkeypatch.setattr(
        module,
        "process_dataset_metadata",
        lambda dataset, metadata_path, output_path, sample_ids=None: calls.append(("metadata", sample_ids)),
    )
    monkeypatch.setattr(module, "create_all_items", lambda cfg, kernel_size, overwrite: calls.append(("items",)))
    monkeypatch.setattr(module, "compute_all_tile_stats", lambda cfg, cell_type_col, overwrite: calls.append(("stats",)))
    monkeypatch.setattr(
        module,
        "create_filtered_items",
        lambda cfg, source_items_name="all", overwrite=False: calls.append(("filtered",)),
    )

    with pytest.raises(RuntimeError, match=r"Failed samples: \['S1'\]"):
        processing_cfg = load_processing_config(config_path)
        module.main(processing_cfg=processing_cfg, overwrite=False, executor="ray")

    assert ("metadata", ["S1", "S2"]) not in calls
    assert ("items",) not in calls
    assert ("stats",) not in calls
    assert ("filtered",) not in calls
