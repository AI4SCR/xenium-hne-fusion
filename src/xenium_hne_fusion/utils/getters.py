from __future__ import annotations

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
class PipelineConfig:
    metadata_csv: Path
    raw_dir: Path
    structured_dir: Path
    processed_dir: Path
    tile_sizes: list[int]
    tile_mpp: float
    filter: FilterConfig = field(default_factory=FilterConfig)


def load_config(path: Path) -> PipelineConfig:
    """Load pipeline config from YAML. Fails fast on missing required keys."""
    data = yaml.safe_load(path.read_text())
    f = data.get("filter", {}) or {}
    return PipelineConfig(
        metadata_csv=Path(data["metadata_csv"]),
        raw_dir=Path(data["raw_dir"]),
        structured_dir=Path(data["structured_dir"]),
        processed_dir=Path(data["processed_dir"]),
        tile_sizes=data["tile_sizes"],
        tile_mpp=data["tile_mpp"],
        filter=FilterConfig(
            organ=f.get("organ"),
            disease_type=f.get("disease_type"),
            species=f.get("species"),
            sample_ids=f.get("sample_ids"),
        ),
    )


def resolve_samples(cfg: PipelineConfig) -> list[str]:
    """
    Filter HEST_v1_3_0.csv by cfg.filter spec.

    cfg.filter.sample_ids short-circuits all other filters when set.
    Raises ValueError if no samples match.

    Returns sorted list of sample_id strings.
    """
    if cfg.filter.sample_ids is not None:
        return sorted(cfg.filter.sample_ids)

    meta = pd.read_csv(cfg.metadata_csv)
    mask = meta.platform == "Xenium"

    if cfg.filter.species:
        mask &= meta.species == cfg.filter.species
    if cfg.filter.organ:
        organs = [cfg.filter.organ] if isinstance(cfg.filter.organ, str) else cfg.filter.organ
        mask &= meta.organ.isin(organs)
    if cfg.filter.disease_type:
        mask &= meta.disease_type == cfg.filter.disease_type

    samples = sorted(meta.loc[mask, "id"].tolist())
    if not samples:
        raise ValueError(f"No Xenium samples match filter: {cfg.filter}")
    return samples
