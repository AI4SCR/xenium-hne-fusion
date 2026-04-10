#!/usr/bin/env python3

import sys
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

from xenium_hne_fusion.utils.getters import build_pipeline_config, load_data_config, resolve_samples

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()
logger.remove()
logger.add(sys.stderr)


def main(config: Path) -> None:
    data_cfg = load_data_config(config)
    cfg = build_pipeline_config(data_cfg)

    if data_cfg.name == "hest1k":
        from scripts.data.run_hest1k import filter_hest_samples_by_tile_mpp, get_hest_metadata_path

        metadata_path = get_hest_metadata_path(cfg.raw_dir)
        sample_ids = resolve_samples(cfg, metadata_path)
        sample_ids = filter_hest_samples_by_tile_mpp(cfg, sample_ids, metadata_path)
    elif data_cfg.name == "beat":
        from scripts.data.run_beat import resolve_beat_samples

        sample_ids = resolve_beat_samples(cfg)
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
