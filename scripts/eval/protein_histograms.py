"""
Plot per-protein histograms of cell-level protein expression, split by cancer type.

Reads whole-sample, per-cell protein intensities from `01_structured/<name>/<sample_id>/proteins.parquet`
(pre-tiling, i.e. not aggregated over tiles). Samples are grouped into cancer types by the second
underscore-separated token of their sample ID (e.g. `CH_C_518a_x2` -> `C`). For each protein in the
panel, one figure is saved with an overlaid histogram per cancer type, pooling all cells across all
samples of that type.

Output layout: `<output_dir>/figures/protein_histograms/<protein>.png`

Usage:
    uv run python scripts/eval/protein_histograms.py \\
        --histograms.name owkin \\
        --histograms.data_dir $DATA_DIR
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from xenium_hne_fusion.targets import PROTEIN_PANEL, load_sample_proteins
from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class HistogramsConfig:
    name: str = None
    data_dir: Path = None
    proteins: list[str] = field(default_factory=lambda: list(PROTEIN_PANEL))
    bins: int = 100


def sample_cancer_type(sample_id: str) -> str:
    _, cancer_type, *_ = sample_id.split("_")
    return cancer_type


def load_cell_proteins(structured_dir: Path, proteins: list[str]) -> pd.DataFrame:
    sample_dirs = sorted(p for p in structured_dir.iterdir() if p.is_dir())
    assert sample_dirs, f"no sample directories under {structured_dir}"

    frames = []
    for sample_dir in sample_dirs:
        df = load_sample_proteins(structured_dir, sample_dir.name, proteins)
        df["cancer_type"] = sample_cancer_type(sample_dir.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_protein_histogram(df: pd.DataFrame, protein: str, bins: int, out_path: Path) -> None:
    cancer_types = sorted(df["cancer_type"].unique())
    palette = dict(zip(cancer_types, sns.color_palette("colorblind", len(cancer_types))))

    _, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=df, x=protein, hue="cancer_type", hue_order=cancer_types, palette=palette,
        element="step", stat="density", common_norm=False, log_scale=True, bins=bins, ax=ax,
    )
    ax.set_title(protein)
    ax.set_xlabel("cell-level intensity (log scale)")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def main(cfg: HistogramsConfig) -> int:
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    df = load_cell_proteins(managed.structured_dir, cfg.proteins)

    out_dir = managed.figures_dir / "protein_histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    for protein in cfg.proteins:
        plot_protein_histogram(df, protein, cfg.bins, out_dir / f"{protein}.png")

    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(HistogramsConfig, nested_key="histograms")
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.histograms)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
