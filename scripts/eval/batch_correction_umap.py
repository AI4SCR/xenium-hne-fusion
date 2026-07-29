"""
UMAP the corrected embeddings from scripts/eval/batch_correction.py, colored by sample_id and
by cell_type.

Reads the already-written `<out_dir>/<run_name>.zarr` (no recomputation of any correction
method). Draws ONE shared random subsample of cells, reused across every method's panel so
plots show the identical cells (a fair visual comparison of how well each method mixes samples
and separates cell types), fits a 2D UMAP once per method, and plots it twice: colored by
`sample_id` (batch mixing) and by `cell_type` (biological signal, mapped via
`cell_types/owkin/cell_types.json` in batch_correction.py -- consolidates per-sample tumor
labels like "Tu_CH_C_518" into "tumor").

Output: `<output_dir>/figures/batch_correction_umap/<run_name>/<method>_{sample_id,cell_type}.png`,
one pair per method actually present in the zarr (uncorrected, harmony, combat, scanorama,
adtnorm -- whichever were computed). `cell_type` coloring is skipped if the zarr predates that
column (older runs of batch_correction.py).

Usage:
    uv run python scripts/eval/batch_correction_umap.py \\
        --batch_correction_umap.name owkin \\
        --batch_correction_umap.data_dir $DATA_DIR \\
        --batch_correction_umap.run_name c_cells \\
        --batch_correction_umap.debug true
"""

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from xenium_hne_fusion.utils.getters import ManagedPaths

CANDIDATE_METHODS = [
    ("uncorrected", "X_pca"),
    ("harmony", "X_pca_harmony"),
    ("combat", "X_pca_combat"),
    ("scanorama", "X_scanorama"),
    ("adtnorm", "X_pca_adtnorm"),
]


@dataclass
class BatchCorrectionUmapConfig:
    name: str = None
    data_dir: Path = None
    run_name: str = "c_cells"
    n_neighbors: int = 15
    min_dist: float = 0.1
    sample_size: int = 20_000
    random_state: int = 0
    debug: bool = False
    debug_sample_size: int = 2_000


def fit_umap(embedding: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(embedding)


def categorical_palette(n: int) -> list:
    # husl spreads n hues evenly, which becomes hard to distinguish by eye above ~10 categories
    # (e.g. cell_type, often 20+); tab20 varies lightness/saturation too, staying legible further.
    return sns.color_palette("tab20", n) if n > 10 else sns.color_palette("husl", n)


def plot_method_umap(embedding_2d: np.ndarray, labels: pd.Series, label_name: str, method: str, out_path: Path) -> None:
    categories = sorted(labels.unique())
    palette = dict(zip(categories, categorical_palette(len(categories))))

    _, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels.to_numpy(), hue_order=categories,
        palette=palette, s=8, linewidth=0, alpha=0.7, ax=ax,
    )
    ax.set_title(f"{method} -- UMAP colored by {label_name}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title=label_name, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2, fontsize="small")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def main(cfg: BatchCorrectionUmapConfig) -> int:
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    zarr_path = managed.anndata_dir / "batch_correction" / f"{cfg.run_name}.zarr"
    assert zarr_path.exists(), f"{zarr_path} not found -- run scripts/eval/batch_correction.py first"
    adata = ad.read_zarr(zarr_path)

    sample_size = cfg.debug_sample_size if cfg.debug else cfg.sample_size
    sample_size = min(sample_size, adata.n_obs)
    idx = np.random.default_rng(cfg.random_state).choice(adata.n_obs, size=sample_size, replace=False)
    sample_ids = adata.obs["sample_id"].iloc[idx]
    cell_types = adata.obs["cell_type"].iloc[idx] if "cell_type" in adata.obs.columns else None
    logger.info(f"Subsampled {sample_size}/{adata.n_obs} cells, shared across all methods")

    out_dir = managed.figures_dir / "batch_correction_umap" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for method, obsm_key in CANDIDATE_METHODS:
        if obsm_key not in adata.obsm:
            continue
        logger.info(f"Fitting UMAP for {method}...")
        embedding_2d = fit_umap(adata.obsm[obsm_key][idx], cfg.n_neighbors, cfg.min_dist, cfg.random_state)
        plot_method_umap(embedding_2d, sample_ids, "sample_id", method, out_dir / f"{method}_sample_id.png")
        if cell_types is not None:
            plot_method_umap(embedding_2d, cell_types, "cell_type", method, out_dir / f"{method}_cell_type.png")

    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(BatchCorrectionUmapConfig, nested_key="batch_correction_umap")
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.batch_correction_umap)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
