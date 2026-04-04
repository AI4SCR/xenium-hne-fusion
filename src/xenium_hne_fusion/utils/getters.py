from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class FilterConfig:
    organ: str | list[str] | None = None
    disease_type: str | None = None
    species: str | None = None
    sample_ids: list[str] | None = None  # overrides all other filters when set


@dataclass
class DatasetConfig:
    tile_px: int
    stride_px: int
    tile_mpp: float
    filter: FilterConfig = field(default_factory=FilterConfig)


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load dataset processing config from YAML. Paths come from .env, not the YAML."""
    data = yaml.safe_load(path.read_text())
    f = data.get('filter', {}) or {}
    return DatasetConfig(
        tile_px=data['tile_px'],
        stride_px=data['stride_px'],
        tile_mpp=data['tile_mpp'],
        filter=FilterConfig(
            organ=f.get('organ'),
            disease_type=f.get('disease_type'),
            species=f.get('species'),
            sample_ids=f.get('sample_ids'),
        ),
    )


def get_dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    """
    Resolve (download_dir, raw_dir, processed_dir) for a dataset from env vars.

    Expected env vars:
        {DATASET}_DOWNLOAD_DIR
        {DATASET}_RAW_DIR
        {DATASET}_PROCESSED_DIR

    where DATASET is the uppercase dataset name, e.g. HEST1K or BEAT.
    """
    key = dataset.upper()
    download_dir = _require_env(f'{key}_DOWNLOAD_DIR')
    raw_dir = _require_env(f'{key}_RAW_DIR')
    processed_dir = _require_env(f'{key}_PROCESSED_DIR')
    return Path(download_dir), Path(raw_dir), Path(processed_dir)


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    assert val, f'Environment variable {var!r} is not set. Add it to .env.'
    return val


def resolve_samples(cfg: DatasetConfig, metadata_csv: Path) -> list[str]:
    """
    Filter metadata CSV by cfg.filter spec.

    cfg.filter.sample_ids short-circuits all other filters when set.
    Raises ValueError if no samples match.

    Returns sorted list of sample_id strings.
    """
    if cfg.filter.sample_ids is not None:
        return sorted(cfg.filter.sample_ids)

    meta = pd.read_csv(metadata_csv)
    mask = meta.platform == 'Xenium'

    if cfg.filter.species:
        mask &= meta.species == cfg.filter.species
    if cfg.filter.organ:
        organs = [cfg.filter.organ] if isinstance(cfg.filter.organ, str) else cfg.filter.organ
        mask &= meta.organ.isin(organs)
    if cfg.filter.disease_type:
        mask &= meta.disease_type == cfg.filter.disease_type

    samples = sorted(meta.loc[mask, 'id'].tolist())
    if not samples:
        raise ValueError(f'No Xenium samples match filter: {cfg.filter}')
    return samples
