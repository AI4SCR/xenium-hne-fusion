from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.utils import load_panel_config, resolve_training_paths, validate_task_config

load_dotenv()

config_path = Path('configs/train/beat/cell_types/early-fusion.yaml')
# config_path = Path('configs/train/beat/expression/early-fusion.yaml')
cfg = Config.from_yaml(config_path)
cfg.data.items_path = 'all.json'
cfg.data.cache_dir = Path(f'/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/cache/{cfg.task.target}/default')
cfg, _ = resolve_training_paths(cfg)
cfg = load_panel_config(cfg)
validate_task_config(cfg)

dataset_kws = dict(
    target=cfg.task.target,
    items_path=cfg.data.items_path,
    metadata_path=cfg.data.metadata_path,
    source_panel=cfg.data.source_panel,
    target_panel=cfg.data.target_panel if cfg.task.target == "expression" else None,
    include_image=True,
    include_expr=True,
    target_transform=None,
    image_transform=None,
    expr_transform=None,
    expr_pool='token',
    cache_dir=cfg.data.cache_dir,
    drop_nan_columns=True,
    id_key="id",
)

if __name__ == '__main__':
    ds_all = TileDataset(**dataset_kws)
    ds_all.setup()
