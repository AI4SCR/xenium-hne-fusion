import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_batch_correction_module():
    path = Path("scripts/eval/batch_correction.py").resolve()
    spec = importlib.util.spec_from_file_location("batch_correction_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_batch_correction_module()


def _separated_batches(rng, n_per_batch: int, offset: float) -> tuple[np.ndarray, pd.Series]:
    """Two batches offset along dim 0 -- a strong, easily-detectable batch effect."""
    a = rng.normal(loc=0.0, scale=1.0, size=(n_per_batch, 3))
    b = rng.normal(loc=offset, scale=1.0, size=(n_per_batch, 3))
    embedding = np.concatenate([a, b])
    batch = pd.Series(pd.Categorical(["a"] * n_per_batch + ["b"] * n_per_batch))
    return embedding, batch


def test_batch_asw_lower_for_separated_batches(module):
    rng = np.random.default_rng(0)
    separated, batch = _separated_batches(rng, 100, offset=10.0)
    mixed = rng.normal(size=(200, 3))

    assert module.batch_asw(separated, batch, sample_size=200, random_state=0) < module.batch_asw(mixed, batch, sample_size=200, random_state=0)


def test_batch_knn_entropy_lower_for_separated_batches(module):
    rng = np.random.default_rng(0)
    separated, batch = _separated_batches(rng, 100, offset=10.0)
    mixed = rng.normal(size=(200, 3))

    assert module.batch_knn_entropy(separated, batch, n_neighbors=10) < module.batch_knn_entropy(mixed, batch, n_neighbors=10)


def test_batch_pcr_higher_for_separated_batches(module):
    rng = np.random.default_rng(0)
    separated, batch = _separated_batches(rng, 100, offset=10.0)
    mixed = rng.normal(size=(200, 3))

    assert module.batch_pcr(separated, batch) > module.batch_pcr(mixed, batch)


def test_subsample_per_batch_caps_group_size(module):
    df = pd.DataFrame({"sample_id": ["a"] * 50 + ["b"] * 10, "value": range(60)})
    out = module.subsample_per_batch(df, n=20, random_state=0)

    counts = out["sample_id"].value_counts()
    assert counts["a"] == 20
    assert counts["b"] == 10
