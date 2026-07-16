"""Pre-populate the tile feature cache without running training.

Iterates the full dataset once to fill the cache on disk.
Run before supervised training to avoid cache-miss overhead during the first epoch.

Usage:
    uv run python scripts/artifacts/warmup_cache.py \\
        --config configs/train/owkin/proteins/early-fusion.yaml \\
        --train.data.items_path cells.json \\
        --train.data.panel_path default.yaml \\
        --train.data.cache_dir expression/default
"""
from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.train.config import TrainingConfig
from xenium_hne_fusion.train.supervised import build_dataset_kws
from xenium_hne_fusion.train.utils import resolve_training_config


def main(cfg: TrainingConfig) -> None:
    dataset_kws = build_dataset_kws(resolve_training_config(cfg))
    # warmup cache: no transforms and no pooling — both are applied post-cache-load per split dataset.
    kws = {**dataset_kws, 'target_transform': None, 'image_transform': None, 'expr_transform': None, 'expr_pool': 'token'}
    ds = TileDataset(**kws)
    ds.setup()


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    from jsonargparse import ArgumentParser

    load_dotenv(override=True)

    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True)
    parser.add_class_arguments(TrainingConfig, nested_key="train")

    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    main(init.train)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
