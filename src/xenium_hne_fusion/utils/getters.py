from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    organ: str | list[str] | None = None
    disease_type: str | None = None
    species: str | None = None
    sample_ids: list[str] | None = None  # overrides all other filters when set


@dataclass
class PipelineConfig:
    metadata_csv: Path
    download_dir: Path
    raw_dir: Path
    processed_dir: Path
    tile_sizes: list[int]
    tile_mpp: float
    filter: FilterConfig = field(default_factory=FilterConfig)


def load_config(path: Path) -> PipelineConfig:
    """Load pipeline config from YAML. Fails fast on missing required keys."""
    ...


def resolve_samples(cfg: PipelineConfig) -> list[str]:
    """
    Filter HEST_v1_3_0.csv by cfg.filter spec.

    cfg.filter.sample_ids short-circuits all other filters when set.
    Raises ValueError if no samples match.

    Returns sorted list of sample_id strings (e.g. ["XEN_001", "XEN_042"]).
    """
    ...
