"""Pre-populate the tile feature cache without running training.

Iterates the full dataset once to fill the cache on disk.
Run before supervised training to avoid cache-miss overhead during the first epoch.

Usage:
    uv run python scripts/artifacts/warmup_cache.py \\
        --config configs/train/beat/expression/early-fusion.yaml \\
        --data.items_path cells.json \\
        --data.panel_path default.yaml \\
        --data.cache_dir expression/default
"""
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.supervised import build_supervised_dataset_kws
from xenium_hne_fusion.train.utils import prepare_training_config


def main(cfg: Config) -> None:
    dataset_kws = build_supervised_dataset_kws(prepare_training_config(cfg))
    # warmup cache: no transforms and no pooling — both are applied post-cache-load per split dataset.
    kws = {**dataset_kws, 'target_transform': None, 'image_transform': None, 'expr_transform': None, 'expr_pool': 'token'}
    ds = TileDataset(**kws)
    ds.setup()


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Config, None)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)
    raise SystemExit(main(Config(**d)))