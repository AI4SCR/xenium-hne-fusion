"""Shared utilities for structuring raw samples into 01_structured layout."""

from pathlib import Path

from loguru import logger

from xenium_hne_fusion.metadata import link_structured_metadata
from xenium_hne_fusion.tiling import save_points_overview, save_sample_overview


def symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def structure_sample(
    sample_id: str,
    wsi_path: Path,
    transcripts_path: Path,
    structured_dir: Path,
    cells_path: Path | None = None,
) -> None:
    """Create canonical sample dir with symlinked sample artifacts and visualizations."""
    out = structured_dir / sample_id
    out.mkdir(parents=True, exist_ok=True)
    symlink(wsi_path, out / "wsi.tiff")
    symlink(transcripts_path, out / "transcripts.parquet")
    if cells_path is not None:
        symlink(cells_path, out / "cells.parquet")
    save_sample_overview(out / "wsi.tiff", out / "transcripts.parquet", out)
    if cells_path is not None:
        save_points_overview(out / "wsi.tiff", out / "cells.parquet", out / "cells.png")
    logger.info(f"Structured {sample_id} → {out}")


def structure_metadata(metadata_path: Path, structured_dir: Path) -> None:
    dst = link_structured_metadata(metadata_path, structured_dir)
    logger.info(f"Structured metadata → {dst}")
