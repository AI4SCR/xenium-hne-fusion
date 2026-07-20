"""Config definitions for the artifact-creation pipeline stages (items, stats, filter, split, panel)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ItemsThresholdConfig:
    organs: list[str] | None = None
    include_ids: list[str] | None = None
    exclude_ids: list[str] | None = None
    num_transcripts: int | None = None
    num_unique_transcripts: int | None = None
    num_cells: int | None = None
    num_unique_cells: int | None = None


@dataclass
class ItemsFilterConfig:
    name: str
    filter: ItemsThresholdConfig = field(default_factory=ItemsThresholdConfig)


@dataclass
class SplitConfig:
    name: str
    path: Path


@dataclass
class PanelConfig:
    name: str


@dataclass
class ArtifactsConfig:
    name: str
    data_dir: Path
    cell_type_col: str
    tile_px: int
    stride_px: int
    items: ItemsFilterConfig = field(default_factory=lambda: ItemsFilterConfig(name='default'))
    split: SplitConfig = field(default_factory=lambda: SplitConfig(name='default', path=Path('splits/owkin.yaml')))
    panel: PanelConfig | None = None


def build_artifacts_parser():
    """Base parser shared by `scripts/artifacts/*.py` entrypoints: `--config`, `artifacts.*`, `--overwrite`.

    Callers add any script-specific flags (e.g. `--batch-size`) before parsing.
    """
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config', action='config', required=True)
    parser.add_class_arguments(ArtifactsConfig, nested_key='artifacts')
    parser.add_argument('--overwrite', type=bool, default=False)
    return parser
