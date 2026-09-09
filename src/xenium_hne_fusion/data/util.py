"""Shared utilities for the data pipeline stages (structure, process)."""

from pathlib import Path

from loguru import logger


def to_pyramidal(
    img_path: Path,
    save_path: Path,
    tile: int = 512,
    compression: str = "deflate",
    bigtiff: bool = True,
    **kwargs,
) -> None:
    """Convert a flat TIFF to a tiled, pyramidal TIFF.

    Opens with `access="sequential"` so libvips streams the source top-to-bottom in a
    single pass, bounding memory to a few tile-rows instead of loading the full WSI.
    Requires `libvips` on `LD_LIBRARY_PATH` (set in shell profile, not `.env` — dotenv
    loads too late to affect dlopen's search path).
    """
    import pyvips

    img = pyvips.Image.new_from_file(str(img_path), access="sequential")
    img.tiffsave(
        str(save_path),
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=True,
        compression=compression,  # "jpeg" needed for QuPath compatibility
        bigtiff=bigtiff,
        **kwargs,
    )
    logger.info(f"Saved pyramidal TIFF to: {save_path}")


def is_pyramidal_tiff(path: Path) -> bool:
    """Check whether `path` is already a tiled, multi-resolution (pyramidal) TIFF.

    Opens with `is_ome=False`: some OME-TIFFs (e.g. per-channel Xenium morphology images)
    declare sibling files as companions in their OME-XML and get stitched into a bogus
    multi-file series by tifffile's default OME parsing.
    """
    import tifffile

    with tifffile.TiffFile(path, is_ome=False) as tf:
        page = tf.pages[0]
        return bool(page.is_tiled) and len(tf.series[0].levels) > 1
