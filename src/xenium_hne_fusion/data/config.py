"""Config definition for the data pipeline stages (structure, process).

`DataConfig` carries plain identifying fields (`name`, `data_dir`, `raw_dir`), not a
`ManagedPaths` object. Scripts construct `ManagedPaths` independently from these
fields — see `xenium_hne_fusion.utils.getters.ManagedPaths`.
"""

from dataclasses import dataclass, field
from pathlib import Path

from xenium_hne_fusion.config import FilterConfig, TilesConfig


@dataclass
class DataConfig:
    """Dataset definition shared by the structure and process pipeline scripts."""

    name: str
    data_dir: Path
    raw_dir: Path
    cell_type_col: str
    cell_types_path: Path
    tiles: TilesConfig
    filter: FilterConfig = field(default_factory=FilterConfig)
