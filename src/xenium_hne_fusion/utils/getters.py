import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from xenium_hne_fusion.config import (
    ArtifactsConfig,
    DataConfig,
    FilterConfig,
    ItemsConfig,
    ItemsThresholdConfig,
    PanelConfig,
    SplitConfig,
    TilesConfig,
)
from xenium_hne_fusion.metadata import normalize_hest1k_metadata, normalize_sample_metadata, read_metadata_table


def get_repo_root() -> Path:
    env_root = os.environ.get('XHF_REPO_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    root = Path(__file__).resolve().parents[3]
    assert (root / 'pyproject.toml').exists(), f'Could not resolve repo root from {__file__}'
    return root


@dataclass(frozen=True)
class ManagedPaths:
    data_dir: Path
    structured_dir: Path
    processed_dir: Path
    output_dir: Path


@dataclass
class PipelineConfig:
    dataset: str
    raw_dir: Path
    paths: ManagedPaths
    data: DataConfig

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def structured_dir(self) -> Path:
        return self.paths.structured_dir

    @property
    def processed_dir(self) -> Path:
        return self.paths.processed_dir

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir

    @property
    def tile_px(self) -> int:
        return self.data.tiles.tile_px

    @property
    def stride_px(self) -> int:
        return self.data.tiles.stride_px

    @property
    def tile_mpp(self) -> float:
        return self.data.tiles.mpp

    @property
    def kernel_size(self) -> int:
        return self.data.tiles.kernel_size

    @property
    def predicate(self) -> str:
        return self.data.tiles.predicate

    @property
    def filter(self) -> FilterConfig:
        return self.data.filter


def load_data_config(path: Path) -> DataConfig:
    data = yaml.safe_load(path.read_text()) or {}
    tiles = data.get('tiles') or {
        'tile_px': data.get('tile_px'),
        'stride_px': data.get('stride_px'),
        'mpp': data.get('tile_mpp'),
        'kernel_size': data.get('kernel_size', 16),
        'img_size': data.get('img_size'),
        'predicate': data.get('predicate', 'within'),
    }
    assert tiles.get('img_size') is not None, f"Missing tiles.img_size in {path}"
    filter_data = data.get('filter') or {}
    return DataConfig(
        name=data['name'],
        tiles=TilesConfig(
            tile_px=tiles['tile_px'],
            stride_px=tiles['stride_px'],
            mpp=tiles['mpp'],
            kernel_size=tiles.get('kernel_size', 16),
            img_size=tiles.get('img_size'),
            predicate=tiles.get('predicate', 'within'),
        ),
        filter=FilterConfig(
            organ=filter_data.get('organ'),
            disease_type=filter_data.get('disease_type'),
            species=filter_data.get('species'),
            include_ids=filter_data.get('include_ids'),
            exclude_ids=filter_data.get('exclude_ids'),
        ),
    )


def load_artifacts_config(path: Path) -> ArtifactsConfig:
    data = yaml.safe_load(path.read_text()) or {}
    items_data = data.get('items') or {'name': 'default', 'filter': {}}
    items_filter_data = items_data.get('filter') or {}
    split_data = data.get('split') or {'name': 'default', 'test_size': 0.25, 'val_size': 0.25}
    panel_data = data.get('panel')
    return ArtifactsConfig(
        name=data['name'],
        items=ItemsConfig(
            name=items_data['name'],
            filter=ItemsThresholdConfig(
                organs=items_filter_data.get('organs'),
                include_ids=items_filter_data.get('include_ids'),
                exclude_ids=items_filter_data.get('exclude_ids'),
                num_transcripts=items_filter_data.get('num_transcripts'),
                num_unique_transcripts=items_filter_data.get('num_unique_transcripts'),
                num_cells=items_filter_data.get('num_cells'),
                num_unique_cells=items_filter_data.get('num_unique_cells'),
            ),
        ),
        split=SplitConfig(
            name=split_data['name'],
            test_size=split_data.get('test_size'),
            val_size=split_data.get('val_size'),
            stratify=split_data.get('stratify', False),
            target_column_name=split_data.get('target_column_name'),
            encode_targets=split_data.get('encode_targets', False),
            nan_value=split_data.get('nan_value', -1),
            use_filtered_targets_for_train=split_data.get('use_filtered_targets_for_train', False),
            include_targets=split_data.get('include_targets'),
            group_column_name=split_data.get('group_column_name'),
            random_state=split_data.get('random_state'),
        ),
        panel=None if panel_data is None else PanelConfig(
            name=panel_data.get('name'),
            n_top_genes=panel_data.get('n_top_genes'),
            flavor=panel_data.get('flavor'),
        ),
    )


def load_dataset_config(path: Path) -> DataConfig:
    return load_data_config(path)


def get_data_config_path(dataset: str, config_root: Path | None = None) -> Path:
    root = config_root or Path('configs/data/local')
    return root / f'{dataset}.yaml'


def get_artifacts_config_path(dataset: str, config_root: Path | None = None) -> Path:
    root = config_root or Path('configs/artifacts')
    return root / dataset / 'default.yaml'


load_processing_config = load_data_config
get_processing_config_path = get_data_config_path


def get_data_dir() -> Path:
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
    return get_managed_paths(name).output_dir / 'panels'


def validate_filter_ids(filter_cfg: FilterConfig) -> None:
    assert filter_cfg.include_ids is None or filter_cfg.exclude_ids is None, 'include_ids and exclude_ids are mutually exclusive'


def select_sample_ids(
    available_sample_ids: list[str],
    filter_cfg: FilterConfig,
) -> list[str]:
    validate_filter_ids(filter_cfg)
    available = sorted(available_sample_ids)
    available_set = set(available)

    if filter_cfg.include_ids is not None:
        missing = sorted(set(filter_cfg.include_ids) - available_set)
        assert not missing, f'Unknown sample_ids in include_ids: {missing}'
        selected = sorted(filter_cfg.include_ids)
    elif filter_cfg.exclude_ids is not None:
        missing = sorted(set(filter_cfg.exclude_ids) - available_set)
        assert not missing, f'Unknown sample_ids in exclude_ids: {missing}'
        selected = [sample_id for sample_id in available if sample_id not in set(filter_cfg.exclude_ids)]
    else:
        selected = available

    assert selected, f'No Xenium samples match filter: {filter_cfg}'
    return selected


def build_pipeline_config(cfg: DataConfig) -> PipelineConfig:
    dataset = cfg.name
    assert dataset in {'beat', 'hest1k'}, f'Unknown dataset for config name: {cfg.name!r}'
    managed = get_managed_paths(cfg.name)
    raw_dir = _require_env_path(f'{dataset.upper()}_RAW_DIR')
    return PipelineConfig(
        dataset=dataset,
        raw_dir=raw_dir,
        paths=managed,
        data=cfg,
    )


def load_pipeline_config(
    dataset: str | None = None,
    config_path: Path | None = None,
    config_root: Path | None = None,
) -> PipelineConfig:
    assert dataset is not None or config_path is not None, 'dataset or config_path is required'
    if config_path is None:
        config_path = get_data_config_path(dataset, config_root=config_root)
    cfg = load_data_config(config_path)
    if dataset is not None:
        assert cfg.name in {'beat', 'hest1k'}, f'Unknown dataset for config name: {cfg.name!r}'
        assert dataset == cfg.name, f"Config name {cfg.name!r} does not match dataset {dataset!r}"
    return build_pipeline_config(cfg)


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


DEFAULT_CELL_TYPE_COL = 'Level3_grouped'
DEFAULT_SOURCE_ITEMS_NAME = 'all'
STAT_COLS = ['num_transcripts', 'num_unique_transcripts', 'num_cells', 'num_unique_cells']


def structured_done_path(cfg: 'PipelineConfig', sample_id: str) -> Path:
    return cfg.paths.structured_dir / sample_id / '.structured.done'


def processed_done_path(cfg: 'PipelineConfig', sample_id: str) -> Path:
    tiles = cfg.data.tiles
    return cfg.paths.processed_dir / sample_id / f'{tiles.tile_px}_{tiles.stride_px}' / '.processed.done'


def processed_sample_dir(cfg: 'PipelineConfig', sample_id: str) -> Path:
    tiles = cfg.data.tiles
    return cfg.paths.processed_dir / sample_id / f'{tiles.tile_px}_{tiles.stride_px}'


def is_sample_structured(cfg: 'PipelineConfig', sample_id: str) -> bool:
    return structured_done_path(cfg, sample_id).exists()


def is_sample_processed(cfg: 'PipelineConfig', sample_id: str) -> bool:
    return processed_done_path(cfg, sample_id).exists()


def mark_sample_structured(cfg: 'PipelineConfig', sample_id: str) -> None:
    path = structured_done_path(cfg, sample_id)
    assert path.parent.exists(), f'Structured sample dir missing: {path.parent}'
    path.write_text(f'sample_id={sample_id}\nstage=structured\n')


def mark_sample_processed(cfg: 'PipelineConfig', sample_id: str) -> None:
    path = processed_done_path(cfg, sample_id)
    assert path.parent.exists(), f'Processed sample dir missing: {path.parent}'
    path.write_text(f'sample_id={sample_id}\nstage=processed\n')


def clear_sample_markers(cfg: 'PipelineConfig', sample_id: str) -> None:
    structured_done_path(cfg, sample_id).unlink(missing_ok=True)
    processed_done_path(cfg, sample_id).unlink(missing_ok=True)


def tile_item(tile_dir: Path, sample_id: str, tile_id: int, kernel_size: int = 16) -> dict | None:
    if not (tile_dir / 'tile.pt').exists():
        return None
    if not (tile_dir / f'expr-kernel_size={kernel_size}.parquet').exists():
        return None
    if not (tile_dir / 'transcripts.parquet').exists():
        return None
    return {
        'id': f'{sample_id}_{tile_id}',
        'sample_id': sample_id,
        'tile_id': tile_id,
        'tile_dir': str(tile_dir),
    }


def iter_tile_dirs(sample_dir: Path) -> list[Path]:
    direct_tile_dirs = [p for p in sample_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if direct_tile_dirs:
        return sorted(direct_tile_dirs, key=lambda p: int(p.name))
    config_dirs = [p for p in sample_dir.iterdir() if p.is_dir()]
    assert len(config_dirs) == 1, f'Expected exactly one tile-config dir in {sample_dir}, found {config_dirs}'
    tile_root = config_dirs[0]
    return sorted(
        [p for p in tile_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def compute_item_stats(item: dict, cell_type_col: str) -> dict:
    tile_dir = Path(item['tile_dir'])
    transcripts = pd.read_parquet(tile_dir / 'transcripts.parquet', columns=['feature_name'])

    num_cells = float('nan')
    num_unique_cells = float('nan')
    cells_path = tile_dir / 'cells.parquet'
    if cells_path.exists():
        cells = pd.read_parquet(cells_path, columns=[cell_type_col])
        num_cells = len(cells)
        num_unique_cells = cells[cell_type_col].nunique()

    return {
        'id': item['id'],
        'num_transcripts': len(transcripts),
        'num_unique_transcripts': transcripts['feature_name'].nunique(),
        'num_cells': num_cells,
        'num_unique_cells': num_unique_cells,
    }


def apply_filter(stats: pd.DataFrame, cfg: ItemsConfig) -> pd.Series:
    mask = pd.Series(True, index=stats.index)
    for field in STAT_COLS:
        threshold = getattr(cfg.filter, field)
        if threshold is None:
            continue
        mask &= stats[field].notna() & (stats[field] >= threshold)
    return mask


def resolve_samples(cfg: DataConfig | PipelineConfig, metadata_path: Path) -> list[str]:
    filter_cfg = cfg.data.filter if isinstance(cfg, PipelineConfig) else cfg.filter
    validate_filter_ids(filter_cfg)

    dataset_name = cfg.data.name if isinstance(cfg, PipelineConfig) else cfg.name
    if dataset_name == 'hest1k':
        meta = normalize_hest1k_metadata(read_metadata_table(metadata_path))
    else:
        meta = normalize_sample_metadata(read_metadata_table(metadata_path))
    mask = pd.Series(True, index=meta.index)

    platform_col = 'platform' if 'platform' in meta.columns else 'st_technology' if 'st_technology' in meta.columns else None
    if platform_col is not None:
        mask &= meta[platform_col] == 'Xenium'

    if filter_cfg.species:
        mask &= meta.species == filter_cfg.species
    if filter_cfg.organ:
        organs = [filter_cfg.organ] if isinstance(filter_cfg.organ, str) else filter_cfg.organ
        mask &= meta.organ.isin(organs)
    if filter_cfg.disease_type:
        disease_col = 'disease_type' if 'disease_type' in meta.columns else 'disease_state'
        mask &= meta[disease_col] == filter_cfg.disease_type

    samples = sorted(meta.loc[mask, 'sample_id'].tolist())
    return select_sample_ids(samples, filter_cfg)
