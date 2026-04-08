from dataclasses import dataclass, field


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
    kernel_size: int = 16
    predicate: str = 'within'


@dataclass
class ItemsThresholdConfig:
    organs: list[str] | None = None
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
    split_name: str
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
class ProcessingConfig:
    name: str
    tiles: TilesConfig
    filter: FilterConfig = field(default_factory=FilterConfig)
    items: ItemsConfig = field(default_factory=lambda: ItemsConfig(name='default'))
    split: SplitConfig = field(default_factory=lambda: SplitConfig(split_name='default', test_size=0.25, val_size=0.25))
