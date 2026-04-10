"""Detect tissue regions for a single sample."""

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import detect_tissues


def main(wsi_path: Path, output_parquet: Path) -> None:
    detect_tissues(wsi_path, output_parquet)


def cli() -> int:
    auto_cli(main)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
