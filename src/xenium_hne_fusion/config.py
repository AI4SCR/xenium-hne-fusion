from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    organ: str | list[str] | None = None
    disease_type: str | None = None
    species: str | None = None
    include_ids: list[str] | None = None
    exclude_ids: list[str] | None = None


@dataclass
class TilesConfig:
    tile_px: int
    stride_px: int
    mpp: float
    img_size: int
    kernel_size: int = 16
    predicate: str = 'within'


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
class ItemsConfig:
    name: str
    filter: ItemsThresholdConfig = field(default_factory=ItemsThresholdConfig)


@dataclass
class SplitConfig:
    name: str
    test_size: float | None = None
    val_size: float | None = None
    stratify: bool = False
    target_column_name: str | None = None
    encode_targets: bool = False
    nan_value: int = -1
    use_filtered_targets_for_train: bool = False
    include_targets: list[str] | None = None
    group_column_name: str | None = None
    random_state: int | None = None


@dataclass
class PanelConfig:
    name: str | None = None
    metadata_path: Path | None = None
    n_top_genes: int | None = None
    flavor: str | None = None


@dataclass
class DataConfig:
    name: str
    tiles: TilesConfig
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class ArtifactsConfig:
    name: str
    items: ItemsConfig = field(default_factory=lambda: ItemsConfig(name='default'))
    split: SplitConfig = field(default_factory=lambda: SplitConfig(name='default', test_size=0.25, val_size=0.25))
    panel: PanelConfig | None = None


@dataclass
class EvalConfig:
    @dataclass
    class Filters:
        target: str       # expression / cell_types
        name: str         # hest1k / beat
        items_path: str   # filename as in train config, e.g. all.json
        metadata_paths: list[str] | None = None
        panel_paths: list[str] | None = None

    project: str
    output_dir: Path
    filters: Filters
    baseline: str = 'vision'
    parameter_columns: list[str] | None = None
    sort_by_score: bool = True


ProcessingConfig = DataConfig
