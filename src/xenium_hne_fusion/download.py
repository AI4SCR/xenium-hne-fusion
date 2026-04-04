from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

from xenium_hne_fusion.metadata import link_structured_metadata
from xenium_hne_fusion.tiling import save_sample_overview


def download_hest_metadata(raw_dir: Path) -> Path:
    """Download the HEST metadata CSV into the dataset raw dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading HEST metadata → {raw_dir}")
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=["HEST_v1_3_0.csv"],
    )
    return raw_dir / "HEST_v1_3_0.csv"


def download_sample(sample_id: str, raw_dir: Path) -> Path:
    """
    Download a single HEST sample from HuggingFace via snapshot_download.

    Downloads wsis/ and transcripts/ for the given sample_id. Idempotent —
    snapshot_download resumes.

    Returns raw_dir (files land at raw_dir/wsis/, raw_dir/transcripts/).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {sample_id} → {raw_dir}")
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=[
            f"wsis/{sample_id}*",
            f"transcripts/{sample_id}*",
        ],
    )
    return raw_dir


def create_structured_symlinks(sample_id: str, raw_dir: Path, structured_dir: Path) -> None:
    """
    Create canonical symlinks under 01_structured/hest1k/<sample_id>/:

        wsi.tiff            → <raw_dir>/wsis/<file>
        transcripts.parquet → <raw_dir>/transcripts/<file>

    Asserts exactly one WSI and one transcript file exist for the sample.
    Skips already-existing symlinks.
    """
    wsi_files = list((raw_dir / "wsis").glob(f"{sample_id}*"))
    tx_files = list((raw_dir / "transcripts").glob(f"{sample_id}*"))
    assert len(wsi_files) == 1, f"Expected 1 WSI for {sample_id}, found: {wsi_files}"
    assert len(tx_files) == 1, f"Expected 1 transcript file for {sample_id}, found: {tx_files}"

    sample_structured = structured_dir / sample_id
    sample_structured.mkdir(parents=True, exist_ok=True)

    _symlink(wsi_files[0], sample_structured / "wsi.tiff")
    _symlink(tx_files[0], sample_structured / "transcripts.parquet")
    save_sample_overview(
        sample_structured / "wsi.tiff",
        sample_structured / "transcripts.parquet",
        sample_structured,
    )
    logger.info(f"Structured symlinks created at {sample_structured}")


def create_structured_metadata_symlink(metadata_path: Path, structured_dir: Path) -> Path:
    dst = link_structured_metadata(metadata_path, structured_dir)
    logger.info(f"Structured metadata symlink created at {dst}")
    return dst


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        return
    dst.symlink_to(src.resolve())
