"""Tile tissue regions for a single sample."""

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import tile_tissues


def main(
    wsi_path: Path,
    tissues_parquet: Path,
    output_parquet: Path,
    tile_px: int = 256,
    stride_px: int = 256,
    mpp: float = 0.5,
    slide_mpp: float | None = None,
) -> None:
    tile_tissues(wsi_path, tissues_parquet, tile_px, stride_px, mpp, output_parquet, slide_mpp=slide_mpp)


def cli() -> int:
    auto_cli(main)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
