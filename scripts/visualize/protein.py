"""
Plot per-protein Xenium morphology channel images with QC-mask overlays, for one sample.

Reads raw per-channel OME-TIFFs directly from the raw dataset root (not the managed
01_structured/02_processed tree, which doesn't carry morphology images):

    <raw_dir>/matched_samples/<sample_id>/xenium/normalised_results/outs/
        morphology_focus/ch<NNNN>_<slug>.ome.tif
        aux_outputs/morphology_focus_qc_masks/ch<NNNN>_<slug>.ome.tif

These are already tiled, pyramidal OME-TIFFs (matching the format `to_pyramidal` produces
for WSIs), so no re-encoding is needed. A symlink into `02_processed/<name>/<sample_id>/`
is created so the managed dataset root has a stable pointer to the channel/QC-mask files
actually used.

`--offset` is subtracted from raw channel intensities before display (default 0, i.e. no
change) -- surfaced as an explicit knob rather than assumed, pending confirmation of a
reported "~100" baseline offset in the raw image data.

Output layout: <output_dir>/figures/protein_morphology/<sample_id>/<protein>.png

Usage:
    uv run python scripts/visualize/protein.py \\
        --viz.name owkin \\
        --viz.data_dir $DATA_DIR \\
        --viz.raw_dir $OWKIN_RAW_DIR \\
        --viz.sample_id CH_C_518a_x2
"""

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from xenium_hne_fusion.protein_morphology import (
    ensure_pyramidal_channel,
    match_protein_files,
    plot_protein_with_qc,
    read_channel_at_level,
    select_level_index,
)
from xenium_hne_fusion.targets import PROTEIN_PANEL
from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class ProteinVizConfig:
    name: str = None
    data_dir: Path = None
    raw_dir: Path = None
    sample_id: str = None
    proteins: list[str] = field(default_factory=lambda: list(PROTEIN_PANEL))
    offset: int = 0
    max_size: int = 2000


def main(cfg: ProteinVizConfig, *, overwrite: bool = False) -> int:
    assert cfg.name and cfg.data_dir and cfg.raw_dir and cfg.sample_id, "name, data_dir, raw_dir, sample_id are required"
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name, raw_dir=cfg.raw_dir)

    raw_outs_dir = cfg.raw_dir / "matched_samples" / cfg.sample_id / "xenium" / "normalised_results" / "outs"
    morphology_dir = raw_outs_dir / "morphology_focus"
    qc_dir = raw_outs_dir / "aux_outputs" / "morphology_focus_qc_masks"
    assert morphology_dir.exists(), f"{morphology_dir} does not exist"
    assert qc_dir.exists(), f"{qc_dir} does not exist"

    protein_to_channel = match_protein_files(morphology_dir, cfg.proteins)
    protein_to_qc = match_protein_files(qc_dir, cfg.proteins)

    processed_morphology_dir = managed.processed_dir / cfg.sample_id / "morphology"
    processed_qc_dir = managed.processed_dir / cfg.sample_id / "morphology_qc_masks"
    out_dir = managed.figures_dir / "protein_morphology" / cfg.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for protein in cfg.proteins:
        out_path = out_dir / f"{protein}.png"
        if out_path.exists() and not overwrite:
            logger.info(f"Skipping already-plotted {protein}")
            continue

        channel_dst = processed_morphology_dir / protein_to_channel[protein].name
        qc_dst = processed_qc_dir / protein_to_qc[protein].name
        ensure_pyramidal_channel(protein_to_channel[protein], channel_dst)
        ensure_pyramidal_channel(protein_to_qc[protein], qc_dst)

        level = select_level_index(channel_dst, cfg.max_size)
        channel_arr = read_channel_at_level(channel_dst, level)
        qc_arr = read_channel_at_level(qc_dst, level)

        logger.info(
            f"{protein}: level {level}, shape {channel_arr.shape}, "
            f"intensity min/mean/max = {channel_arr.min()}/{channel_arr.mean():.1f}/{channel_arr.max()}"
        )
        plot_protein_with_qc(channel_arr, qc_arr, cfg.offset, protein, out_path)

    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(ProteinVizConfig, nested_key="viz")
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.viz, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
