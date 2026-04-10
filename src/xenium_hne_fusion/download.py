
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from loguru import logger
from wsidata import open_wsi

from xenium_hne_fusion.metadata import normalize_hest1k_metadata, read_metadata_table
from xenium_hne_fusion.structure import structure_metadata, structure_sample, symlink


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


def validate_hest_sample_mpp(
    sample_id: str,
    raw_dir: Path,
    metadata_path: Path,
    rel_tol: float = 0.10,
) -> None:
    row = get_hest_metadata_row(sample_id, metadata_path)
    if row is None:
        logger.warning(f"HEST MPP validation skipped for {sample_id}: expected 1 metadata row, found 0")
        return
    expected = row.get("pixel_size_um_estimated")
    if pd.isna(expected):
        logger.warning(f"HEST MPP validation skipped for {sample_id}: missing pixel_size_um_estimated")
        return

    wsi_files = list((raw_dir / "wsis").glob(f"{sample_id}*"))
    if len(wsi_files) != 1:
        logger.warning(f"HEST MPP validation skipped for {sample_id}: expected 1 WSI, found {len(wsi_files)}")
        return

    native_mpp = open_wsi(wsi_files[0]).properties.mpp
    if native_mpp is None:
        logger.warning(f"HEST MPP validation skipped for {sample_id}: WSI has no mpp metadata")
        return

    expected = float(expected)
    native_mpp = float(native_mpp)
    relative_error = abs(native_mpp - expected) / expected

    if relative_error > rel_tol:
        logger.warning(
            f"HEST MPP mismatch for {sample_id}: "
            f"wsi_mpp={native_mpp:.6f}, pixel_size_um_estimated={expected:.6f}, relative_error={relative_error:.1%}"
        )


def get_hest_metadata_row(sample_id: str, metadata_path: Path) -> pd.Series | None:
    metadata = normalize_hest1k_metadata(read_metadata_table(metadata_path))
    rows = metadata.loc[metadata["sample_id"] == sample_id]
    assert len(rows) <= 1, f"Expected at most 1 metadata row for {sample_id}, found {len(rows)}"
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def get_hest_sample_mpp(sample_id: str, metadata_path: Path) -> float:
    row = get_hest_metadata_row(sample_id, metadata_path)
    assert row is not None, f"HEST metadata row not found for {sample_id}"
    mpp = row.get("pixel_size_um_estimated")
    assert not pd.isna(mpp), f"pixel_size_um_estimated missing for {sample_id}"
    return float(mpp)


def create_structured_symlinks(sample_id: str, raw_dir: Path, structured_dir: Path) -> None:
    wsi_files = list((raw_dir / "wsis").glob(f"{sample_id}*"))
    tx_files = list((raw_dir / "transcripts").glob(f"{sample_id}*"))
    assert len(wsi_files) == 1, f"Expected 1 WSI for {sample_id}, found: {wsi_files}"
    assert len(tx_files) == 1, f"Expected 1 transcript file for {sample_id}, found: {tx_files}"
    structure_sample(sample_id, wsi_files[0], tx_files[0], structured_dir)


def create_structured_metadata_symlink(metadata_path: Path, structured_dir: Path) -> Path:
    structure_metadata(metadata_path, structured_dir)
    return structured_dir / f"metadata{metadata_path.suffix}"
