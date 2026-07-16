from dotenv import load_dotenv
assert load_dotenv()

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedPaths:
    """All managed dataset paths, derived from `data_dir` and `name`.

    `raw_dir` is the unmanaged source root for a dataset (only needed by the
    structure stage) and is kept separate from the derived `data_dir` layout.
    """

    data_dir: Path
    name: str
    raw_dir: Path | None = None

    @property
    def structured_dir(self) -> Path:
        return self.data_dir / '01_structured' / self.name

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / '02_processed' / self.name

    @property
    def output_dir(self) -> Path:
        return self.data_dir / '03_output' / self.name

    @property
    def panels_dir(self) -> Path:
        return self.output_dir / 'panels'

    @property
    def items_dir(self) -> Path:
        return self.output_dir / 'items'

    @property
    def statistics_dir(self) -> Path:
        return self.output_dir / 'statistics'

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / 'figures'

    def resolve_items_path(self, path: Path) -> Path:
        path = Path(os.path.expandvars(path))
        return path if path.is_absolute() else self.items_dir / path

    def resolve_statistics_path(self, path: Path) -> Path:
        path = Path(os.path.expandvars(path))
        return path if path.is_absolute() else self.statistics_dir / path


def get_hest_metadata_path(raw_dir: Path) -> Path:
    from xenium_hne_fusion.download import download_hest_metadata
    metadata_path = raw_dir / "HEST_v1_3_0.csv"
    if metadata_path.exists():
        return metadata_path
    return download_hest_metadata(raw_dir)
