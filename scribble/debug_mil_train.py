from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.train.mil import train
from xenium_hne_fusion.train.mil_config import MILConfig

load_dotenv(override=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a MIL config and call the train function directly.")
    parser.add_argument("config", type=Path, help="Path to the MIL YAML config.")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override cfg.debug before calling train().",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = MILConfig.from_yaml(args.config)
    if args.debug is not None:
        cfg.debug = args.debug
    train(cfg, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
