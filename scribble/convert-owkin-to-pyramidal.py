"""

set -euo pipefail

source ~/miniconda3/bin/activate
conda activate /work/FAC/FBM/DBC/mrapsoma/prometex/projects/mesothelioma/workflow-xe-hne/0-prepare-data/.snakemake/conda/724016b4cad7b6c608b9195b55c35428_

set -a
source /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/.env
set + a

for SAMPLE_DIR in $DATA_DIR/01_structured/owkin/CH_*; do

    [ -d "$SAMPLE_DIR" ] || continue

    echo "$SAMPLE_DIR/wsi.tiff->$SAMPLE_DIR/tmp.tiff"
    vips tiffsave $SAMPLE_DIR/wsi.tiff $SAMPLE_DIR/tmp.tiff --tile --pyramid --tile-width 512 --tile-height 512 --compression deflate --bigtiff
    mv $SAMPLE_DIR/tmp.tiff $SAMPLE_DIR/wsi.tiff
done


for SAMPLE_DIR in "$DATA_DIR"/01_structured/owkin/CH_*; do

    [ -d "$SAMPLE_DIR" ] || continue

    SAMPLE_ID="$(basename "$SAMPLE_DIR")"
    echo '${SAMPLE_DIR}/wsi.tiff -> ${SAMPLE_DIR}/tmp.tiff'

    sbatch \
        --job-name="vips_${SAMPLE_ID}" \
        --cpus-per-task=10 \
        --mem=256G \
        --time=04:00:00 \
        --wrap="
            vips tiffsave '${SAMPLE_DIR}/wsi.tiff' '${SAMPLE_DIR}/tmp.tiff' \
                --tile \
                --pyramid \
                --tile-width 512 \
                --tile-height 512 \
                --compression deflate \
                --bigtiff
            mv '${SAMPLE_DIR}/tmp.tiff' '${SAMPLE_DIR}/wsi.tiff'

        "

done
"""


from pathlib import Path

import pyvips
from loguru import logger


def to_pyramidal(
    img_path: Path,
    save_path: Path,
    tile: int = 512,
    compression: str = "deflate",
    bigtiff: bool = True,
    **kwargs,
):
    img = pyvips.Image.new_from_file(img_path, access="sequential")
    img.tiffsave(
        save_path,
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=True,
        compression=compression,
        # "jpeg", "deflate", "zstd", "webp", etc.; 'jpeg' needed for QuPath compatibility
        bigtiff=bigtiff,
        **kwargs,
        # subifd=True,  # if set to true, lazySlides does not recognize the levels.
        # predictor="horizontal",  # useful for deflate/zstd on 16-bit
    )
    logger.info(f"Saved pyramidal TIFF to: {save_path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(to_pyramidal, as_positional=False)
