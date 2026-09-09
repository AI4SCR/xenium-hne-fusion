"""Read and visualize per-channel Xenium morphology images (protein IF) with QC-mask overlays.

Raw morphology output is one OME-TIFF per channel (`ch<NNNN>_<slug>.ome.tif`), tiled and
pyramidal (multi-resolution via SubIFDs), JPEG2000-compressed. `libvips` (used by
`xenium_hne_fusion.data.util.to_pyramidal` for WSIs) cannot decode this JPEG2000 variant —
it silently returns all-zero pixels — so reads here go through `tifffile` instead, and
`ensure_pyramidal_channel` refuses to fall back to a vips conversion.

Each channel's OME-XML declares the full 35-channel panel (shared metadata across sibling
files) and lists the other 34 files as companions via OME's multi-file-series convention.
`tifffile`'s default OME-aware parsing follows that and stitches *all* sibling channel
files in the directory into one bogus (35, H, W) series -- reading "ch0011_cd3e.ome.tif"
this way can silently return another channel's pixels. All reads here open with
`is_ome=False` to force single-file, single-page(+pyramid) parsing instead.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from loguru import logger

from xenium_hne_fusion.data.util import is_pyramidal_tiff
from xenium_hne_fusion.targets import protein_base_name

_CHANNEL_FILENAME = re.compile(r"ch\d+_(.+)\.ome\.tif$")


def match_protein_files(channel_dir: Path, proteins: list[str]) -> dict[str, Path]:
    """Map each `proteins` entry to its raw channel file in `channel_dir`.

    Channel files are named `ch<NNNN>_<slug>.ome.tif`, where `<slug>` is the OME channel
    name lowercased (e.g. "Beta-catenin" -> "beta-catenin", "PD-L1" -> "pd-l1"). Matches
    against `protein_base_name` so panel entries with a "-<digit>" batch suffix (e.g.
    "CD3E-1") resolve to their base channel ("cd3e").
    """
    slug_to_path = {}
    for f in channel_dir.glob("ch*.ome.tif"):
        m = _CHANNEL_FILENAME.match(f.name)
        assert m, f"unexpected channel filename: {f.name}"
        slug_to_path[m.group(1)] = f

    result = {}
    for protein in proteins:
        slug = protein_base_name(protein).lower()
        assert slug in slug_to_path, f"no channel file for {protein!r} (slug {slug!r}) in {channel_dir}"
        result[protein] = slug_to_path[slug]
    return result


def ensure_pyramidal_channel(src_path: Path, dst_path: Path) -> None:
    """Symlink `src_path` at `dst_path` if it's already a pyramidal TIFF.

    Raw morphology/QC-mask channel files are already tiled + pyramidal, so this is
    normally a no-op symlink. There is deliberately no vips-based conversion fallback:
    `to_pyramidal` (libvips) cannot decode the JPEG2000 compression these files use.
    """
    if dst_path.exists() or dst_path.is_symlink():
        return
    assert is_pyramidal_tiff(src_path), (
        f"{src_path} is not a tiled/pyramidal TIFF, and libvips-based conversion isn't "
        "supported here (it cannot decode Xenium's JPEG2000-compressed morphology images)."
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.symlink_to(src_path)


def select_level_index(path: Path, max_size: int) -> int:
    """Pick the smallest pyramid level whose largest dimension is still >= `max_size`."""
    with tifffile.TiffFile(path, is_ome=False) as tf:
        levels = tf.series[0].levels
        for i in range(len(levels) - 1, -1, -1):
            if max(levels[i].shape) >= max_size:
                return i
        return 0


def read_channel_at_level(path: Path, level_index: int) -> np.ndarray:
    with tifffile.TiffFile(path, is_ome=False) as tf:
        return tf.series[0].levels[level_index].asarray()


def plot_protein_with_qc(channel: np.ndarray, qc_mask: np.ndarray, offset: int, title: str, out_path: Path) -> None:
    """Save a grayscale channel image with QC-flagged pixels overlaid in red."""
    assert channel.shape == qc_mask.shape, f"shape mismatch: {channel.shape} vs {qc_mask.shape}"

    img = np.clip(channel.astype(np.float32) - offset, 0, None)
    lo, hi = np.percentile(img, [1, 99.5])
    hi = max(hi, lo + 1)
    img_norm = np.clip((img - lo) / (hi - lo), 0, 1)

    mask_overlay = np.zeros((*qc_mask.shape, 4))
    mask_overlay[qc_mask > 0] = [1, 0, 0, 0.5]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_norm, cmap="gray")
    ax.imshow(mask_overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")
