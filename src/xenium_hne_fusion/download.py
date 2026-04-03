from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger


def download_sample(sample_id: str, download_dir: Path) -> Path:
    """
    Download a single HEST sample from HuggingFace via snapshot_download.

    Downloads wsis/ and transcripts/ for the given sample_id, plus the global
    metadata CSV (HEST_v1_3_0.csv). Idempotent — snapshot_download resumes.

    Returns download_dir (files land at download_dir/wsis/, download_dir/transcripts/).
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {sample_id} → {download_dir}")
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=[
            f"wsis/{sample_id}*",
            f"transcripts/{sample_id}*",
            "HEST_v1_3_0.csv",
        ],
    )
    return download_dir


def create_raw_symlinks(sample_id: str, download_dir: Path, raw_dir: Path) -> None:
    """
    Create canonical symlinks under 01_raw/datasets/hest1k/<sample_id>/:

        wsi.tiff            → <download_dir>/wsis/<file>
        transcripts.parquet → <download_dir>/transcripts/<file>

    Asserts exactly one WSI and one transcript file exist for the sample.
    Skips already-existing symlinks.
    """
    wsi_files = list((download_dir / "wsis").glob(f"{sample_id}*"))
    tx_files = list((download_dir / "transcripts").glob(f"{sample_id}*"))
    assert len(wsi_files) == 1, f"Expected 1 WSI for {sample_id}, found: {wsi_files}"
    assert len(tx_files) == 1, f"Expected 1 transcript file for {sample_id}, found: {tx_files}"

    sample_raw = raw_dir / sample_id
    sample_raw.mkdir(parents=True, exist_ok=True)

    _symlink(wsi_files[0], sample_raw / "wsi.tiff")
    _symlink(tx_files[0], sample_raw / "transcripts.parquet")
    logger.info(f"Raw symlinks created at {sample_raw}")


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        return
    dst.symlink_to(src.resolve())
