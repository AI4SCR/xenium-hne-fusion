#!/usr/bin/env python3

import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

from xenium_hne_fusion.utils.getters import (
    build_pipeline_config,
    filter_hest_samples_by_tile_mpp,
    get_hest_metadata_path,
    load_data_config,
    resolve_beat_samples,
    resolve_hest1k_samples,
    resolve_owkin_samples
)

load_dotenv()
logger.remove()
logger.add(sys.stderr)


def main(config: Path) -> None:
    data_cfg = load_data_config(config)
    cfg = build_pipeline_config(data_cfg)

    if data_cfg.name == "hest1k":
        metadata_path = get_hest_metadata_path(cfg.raw_dir)
        sample_ids = resolve_hest1k_samples(cfg, metadata_path)
        sample_ids = filter_hest_samples_by_tile_mpp(cfg, sample_ids, metadata_path)
    elif data_cfg.name == "beat":
        sample_ids = resolve_beat_samples(cfg)
    elif data_cfg.name == "owkin":
        sample_ids = resolve_owkin_samples(cfg)
    else:
        raise AssertionError(f"Unsupported dataset: {data_cfg.name}")

    print("\n".join(sample_ids))


def cli(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    namespace = parser.parse_args(argv)
    main(namespace.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
