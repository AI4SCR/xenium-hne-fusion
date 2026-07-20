from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    organ: str | list[str] | None = None
    disease_type: str | None = None
    species: str | None = None
    include_ids: list[str] | None = None
    exclude_ids: list[str] | None = None

    def select(self, available_ids: list[str]) -> list[str]:
        assert self.include_ids is None or self.exclude_ids is None, 'include_ids and exclude_ids are mutually exclusive'
        available = sorted(available_ids)
        available_set = set(available)

        if self.include_ids is not None:
            missing = sorted(set(self.include_ids) - available_set)
            assert not missing, f'Unknown sample_ids in include_ids: {missing}'
            selected = sorted(self.include_ids)
        elif self.exclude_ids is not None:
            missing = sorted(set(self.exclude_ids) - available_set)
            assert not missing, f'Unknown sample_ids in exclude_ids: {missing}'
            selected = [sample_id for sample_id in available if sample_id not in set(self.exclude_ids)]
        else:
            selected = available

        assert selected, f'No samples match filter: {self}'
        return selected


@dataclass
class TilesConfig:
    tile_px: int
    stride_px: int
    mpp: float
    img_size: int
    kernel_size: int = 16
    predicate: str = 'within'


@dataclass
class DataConfig:
    name: str
    cell_type_col: str
    tiles: TilesConfig
    filter: FilterConfig = field(default_factory=FilterConfig)
