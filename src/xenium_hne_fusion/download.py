from __future__ import annotations

from pathlib import Path


def download_sample(sample_id: str, download_dir: Path) -> Path:
    """
    Download a single HEST sample from HuggingFace via snapshot_download.

    Downloads only wsis/ and transcripts/ for the given sample_id,
    plus the global metadata CSV (HEST_v1_3_0.csv) if not already present.
    Idempotent — snapshot_download handles resumption natively.

    Returns path to the sample directory (download_dir / sample_id).
    """
    ...


def create_raw_symlinks(sample_id: str, download_dir: Path, raw_dir: Path) -> None:
    """
    Create symlinks in 01_raw/datasets/hest1k/<sample_id>/:

        wsi.tiff            -> <download_dir>/<sample_id>/wsis/<file>.tiff
        transcripts.parquet -> <download_dir>/<sample_id>/transcripts/transcripts.parquet

    Asserts both source files exist before symlinking.
    Does not clobber already-correct symlinks.
    """
    ...
