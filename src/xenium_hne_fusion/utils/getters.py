
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from xenium_hne_fusion.metadata import normalize_sample_metadata, read_metadata_table


def get_repo_root() -> Path:
    env_root = os.environ.get('XHF_REPO_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    root = Path(__file__).resolve().parents[3]
    assert (root / 'pyproject.toml').exists(), f'Could not resolve repo root from {__file__}'
    return root


@dataclass
class FilterConfig:
    organ: str | list[str] | None = None
    disease_type: str | None = None
    species: str | None = None
    sample_ids: list[str] | None = None  # overrides all other filters when set


@dataclass
class DatasetConfig:
    name: str
    tile_px: int
    stride_px: int
    tile_mpp: float
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass(frozen=True)
class ManagedPaths:
    data_dir: Path
    structured_dir: Path
    processed_dir: Path
    output_dir: Path


@dataclass
class PipelineConfig:
    dataset: str
    name: str
    tile_px: int
    stride_px: int
    tile_mpp: float
    raw_dir: Path
    structured_dir: Path
    processed_dir: Path
    output_dir: Path
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class ItemsFilterConfig:
    name: str
    organs: list[str] | None = None
    num_transcripts: int | None = None
    num_unique_transcripts: int | None = None
    num_cells: int | None = None
    num_unique_cells: int | None = None


def load_items_filter_config(path: Path) -> ItemsFilterConfig:
    data = yaml.safe_load(path.read_text())
    return ItemsFilterConfig(
        name=data['name'],
        organs=data.get('organs'),
        num_transcripts=data.get('num_transcripts'),
        num_unique_transcripts=data.get('num_unique_transcripts'),
        num_cells=data.get('num_cells'),
        num_unique_cells=data.get('num_unique_cells'),
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load dataset processing config from YAML. Paths come from .env, not the YAML."""
    data = yaml.safe_load(path.read_text())
    f = data.get('filter', {}) or {}
    return DatasetConfig(
        name=data['name'],
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


def get_dataset_config_path(dataset: str) -> Path:
    return Path('configs/data') / f'{dataset}.yaml'


def get_data_dir() -> Path:
    """Resolve the shared repo-managed data root from $DATA_DIR."""
    return _require_env_path('DATA_DIR')


def get_managed_paths(name: str) -> ManagedPaths:
    data_dir = get_data_dir()
    return ManagedPaths(
        data_dir=data_dir,
        structured_dir=data_dir / '01_structured' / name,
        processed_dir=data_dir / '02_processed' / name,
        output_dir=data_dir / '03_output' / name,
    )


def get_panels_dir(name: str) -> Path:
    return get_repo_root() / 'panels' / name


def load_pipeline_config(dataset: str, config_path: Path | None = None) -> PipelineConfig:
    config_path = config_path or get_dataset_config_path(dataset)
    cfg = load_dataset_config(config_path)
    managed = get_managed_paths(cfg.name)
    raw_dir = _require_env_path(f'{dataset.upper()}_RAW_DIR')
    return PipelineConfig(
        dataset=dataset,
        name=cfg.name,
        tile_px=cfg.tile_px,
        stride_px=cfg.stride_px,
        tile_mpp=cfg.tile_mpp,
        filter=cfg.filter,
        raw_dir=raw_dir,
        structured_dir=managed.structured_dir,
        processed_dir=managed.processed_dir,
        output_dir=managed.output_dir,
    )


def resolve_dataset_paths(dataset: str, name: str) -> tuple[Path, Path, Path, Path]:
    key = dataset.upper()
    raw_dir = _require_env_path(f'{key}_RAW_DIR')
    managed = get_managed_paths(name)
    return raw_dir, managed.structured_dir, managed.processed_dir, managed.output_dir


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    assert val, f'Environment variable {var!r} is not set. Add it to .env.'
    return val


def _require_env_path(var: str) -> Path:
    return Path(_require_env(var)).expanduser().resolve()


def resolve_samples(cfg: DatasetConfig, metadata_path: Path) -> list[str]:
    """
    Filter metadata CSV by cfg.filter spec.

    cfg.filter.sample_ids short-circuits all other filters when set.
    Raises ValueError if no samples match.

    Returns sorted list of sample_id strings.
    """
    if cfg.filter.sample_ids is not None:
        return sorted(cfg.filter.sample_ids)

    meta = normalize_sample_metadata(read_metadata_table(metadata_path))
    mask = pd.Series(True, index=meta.index)

    platform_col = 'platform' if 'platform' in meta.columns else 'st_technology' if 'st_technology' in meta.columns else None
    if platform_col is not None:
        mask &= meta[platform_col] == 'Xenium'

    if cfg.filter.species:
        mask &= meta.species == cfg.filter.species
    if cfg.filter.organ:
        organs = [cfg.filter.organ] if isinstance(cfg.filter.organ, str) else cfg.filter.organ
        mask &= meta.organ.isin(organs)
    if cfg.filter.disease_type:
        disease_col = 'disease_type' if 'disease_type' in meta.columns else 'disease_state'
        mask &= meta[disease_col] == cfg.filter.disease_type

    samples = sorted(meta.loc[mask, 'sample_id'].tolist())
    if not samples:
        raise ValueError(f'No Xenium samples match filter: {cfg.filter}')
    return samples
