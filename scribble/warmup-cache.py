import argparse
from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.utils import _resolve_path, load_panel_config, validate_task_config
from xenium_hne_fusion.utils.getters import get_managed_paths, get_panels_dir

load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--items-path', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--panel-path', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    assert cfg.data.name is not None, 'cfg.data.name must be set'
    cfg.data.items_path = args.items_path
    cfg.data.metadata_path = args.metadata_path
    cfg.data.panel_path = args.panel_path
    cfg.data.cache_dir = args.cache_dir

    output_dir = get_managed_paths(cfg.data.name).output_dir
    cfg.data.items_path = _resolve_path(cfg.data.items_path, root=output_dir / 'items')
    cfg.data.metadata_path = _resolve_path(cfg.data.metadata_path, root=output_dir / 'splits')
    cfg.data.panel_path = _resolve_path(cfg.data.panel_path, root=get_panels_dir(cfg.data.name))
    cfg.data.cache_dir = _resolve_path(cfg.data.cache_dir, root=output_dir / 'cache')

    cfg = load_panel_config(cfg)
    validate_task_config(cfg)

    dataset_kws = dict(
        target=cfg.task.target,
        items_path=cfg.data.items_path,
        metadata_path=cfg.data.metadata_path,
        source_panel=cfg.data.source_panel,
        target_panel=cfg.data.target_panel,
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

    ds_all = TileDataset(**dataset_kws)
    ds_all.setup()
