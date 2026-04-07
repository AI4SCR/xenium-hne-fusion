import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from xenium_hne_fusion.config import FilterConfig, ItemsConfig, ItemsThresholdConfig, ProcessingConfig, SplitConfig, TilesConfig
from xenium_hne_fusion.metadata import normalize_sample_metadata, read_metadata_table


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
    processing: ProcessingConfig

    @property
    def name(self) -> str:
        return self.processing.name

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
        return self.processing.tiles.tile_px

    @property
    def stride_px(self) -> int:
        return self.processing.tiles.stride_px

    @property
    def tile_mpp(self) -> float:
        return self.processing.tiles.mpp

    @property
    def kernel_size(self) -> int:
        return self.processing.tiles.kernel_size

    @property
    def predicate(self) -> str:
        return self.processing.tiles.predicate

    @property
    def filter(self) -> FilterConfig:
        return self.processing.filter

    @property
    def items(self) -> ItemsConfig:
        return self.processing.items

    @property
    def split(self) -> SplitConfig:
        return self.processing.split


@dataclass
class ItemsFilterConfig:
    name: str
    organs: list[str] | None = None
    num_transcripts: int | None = None
    num_unique_transcripts: int | None = None
    num_cells: int | None = None
    num_unique_cells: int | None = None


def load_items_filter_config(path: Path) -> ItemsFilterConfig:
    data = yaml.safe_load(path.read_text()) or {}
    filter_data = data.get('filter', data)
    return ItemsFilterConfig(
        name=data['name'],
        organs=filter_data.get('organs'),
        num_transcripts=filter_data.get('num_transcripts'),
        num_unique_transcripts=filter_data.get('num_unique_transcripts'),
        num_cells=filter_data.get('num_cells'),
        num_unique_cells=filter_data.get('num_unique_cells'),
    )


def load_processing_config(path: Path) -> ProcessingConfig:
    data = yaml.safe_load(path.read_text()) or {}
    tiles = data.get('tiles') or {
        'tile_px': data.get('tile_px'),
        'stride_px': data.get('stride_px'),
        'mpp': data.get('tile_mpp'),
        'kernel_size': data.get('kernel_size', 16),
        'predicate': data.get('predicate', 'within'),
    }
    filter_data = data.get('filter') or {}
    items_data = data.get('items') or {'name': 'default', 'filter': {}}
    items_filter_data = items_data.get('filter') or {}
    split_data = data.get('split') or {'split_name': 'default', 'test_size': 0.25, 'val_size': 0.25}
    return ProcessingConfig(
        name=data['name'],
        tiles=TilesConfig(
            tile_px=tiles['tile_px'],
            stride_px=tiles['stride_px'],
            mpp=tiles['mpp'],
            kernel_size=tiles.get('kernel_size', 16),
            predicate=tiles.get('predicate', 'within'),
        ),
        filter=FilterConfig(
            organ=filter_data.get('organ'),
            disease_type=filter_data.get('disease_type'),
            species=filter_data.get('species'),
            sample_ids=filter_data.get('sample_ids'),
        ),
        items=ItemsConfig(
            name=items_data['name'],
            filter=ItemsThresholdConfig(
                organs=items_filter_data.get('organs'),
                num_transcripts=items_filter_data.get('num_transcripts'),
                num_unique_transcripts=items_filter_data.get('num_unique_transcripts'),
                num_cells=items_filter_data.get('num_cells'),
                num_unique_cells=items_filter_data.get('num_unique_cells'),
            ),
        ),
        split=SplitConfig(
            split_name=split_data['split_name'],
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
    )


def load_dataset_config(path: Path) -> ProcessingConfig:
    return load_processing_config(path)


def get_processing_config_path(dataset: str, config_root: Path | None = None) -> Path:
    root = config_root or Path('configs/data/local')
    return root / f'{dataset}.yaml'


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


def infer_dataset(name: str) -> str:
    dataset = name.split('-', 1)[0]
    assert dataset in {'beat', 'hest1k'}, f'Unknown dataset for config name: {name!r}'
    return dataset


def build_pipeline_config(cfg: ProcessingConfig) -> PipelineConfig:
    dataset = infer_dataset(cfg.name)
    managed = get_managed_paths(cfg.name)
    raw_dir = _require_env_path(f'{dataset.upper()}_RAW_DIR')
    return PipelineConfig(
        dataset=dataset,
        raw_dir=raw_dir,
        paths=managed,
        processing=cfg,
    )


def load_pipeline_config(
    dataset: str | None = None,
    config_path: Path | None = None,
    config_root: Path | None = None,
) -> PipelineConfig:
    assert dataset is not None or config_path is not None, 'dataset or config_path is required'
    if config_path is None:
        config_path = get_processing_config_path(dataset, config_root=config_root)
    cfg = load_processing_config(config_path)
    if dataset is not None:
        assert dataset == infer_dataset(cfg.name), f"Config name {cfg.name!r} does not match dataset {dataset!r}"
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


def resolve_samples(cfg: ProcessingConfig | PipelineConfig, metadata_path: Path) -> list[str]:
    filter_cfg = cfg.processing.filter if isinstance(cfg, PipelineConfig) else cfg.filter

    if filter_cfg.sample_ids is not None:
        return sorted(filter_cfg.sample_ids)

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
    if not samples:
        raise ValueError(f'No Xenium samples match filter: {filter_cfg}')
    return samples
