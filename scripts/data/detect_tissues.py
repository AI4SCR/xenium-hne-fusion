"""Run tissue detection for a single sample."""
from __future__ import annotations

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import detect_tissues


def main(wsi_path: Path, output_parquet: Path) -> None:
    detect_tissues(wsi_path, output_parquet)


if __name__ == "__main__":
    auto_cli(main)
