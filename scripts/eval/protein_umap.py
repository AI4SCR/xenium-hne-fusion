"""
Plot a UMAP embedding of tile-level protein expression, colored by sample ID.

Reads tile-local, per-cell protein intensities from `tile_dir/proteins.parquet` (written by
`tile_cells()` during processing), log1p-transforms and averages across cells within each tile to
get one row per tile over the protein panel, then fits a single UMAP embedding on that
(n_tiles, n_proteins) matrix. The aggregated matrix and embedding are cached so later runs (e.g.
adding per-protein expression coloring) can reuse them without refitting.

Output layout (keyed by the items file stem, so different item sets don't collide):
`<output_dir>/figures/protein_umap/<items_stem>/{tile_proteins,embedding}.parquet`,
`<output_dir>/figures/protein_umap/<items_stem>/sample_id.png`,
`<output_dir>/figures/protein_umap/<items_stem>/<items_stem>_<protein>.png` (one per protein in the panel).

Usage:
    uv run python scripts/eval/protein_umap.py \\
        --umap.name owkin \\
        --umap.data_dir $DATA_DIR \\
        --umap.items_path c_cells.json
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from xenium_hne_fusion.artifacts.items import load_items_dataframe
from xenium_hne_fusion.targets import PROTEIN_PANEL, protein_base_name, protein_base_to_panel
from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class UmapConfig:
    name: str = None
    data_dir: Path = None
    items_path: Path = Path("all.json")
    proteins: list[str] = field(default_factory=lambda: list(PROTEIN_PANEL))
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 0


def aggregate_tile_proteins(tile_dir: Path, base_to_panel: dict[str, str]) -> pd.Series | None:
    proteins_path = tile_dir / "proteins.parquet"
    if not proteins_path.exists():
        return None

    df = pd.read_parquet(proteins_path)
    df = df.drop(columns="geometry", errors="ignore")
    rename = {c: base_to_panel[protein_base_name(c)] for c in df.columns if protein_base_name(c) in base_to_panel}
    df = df.rename(columns=rename)[list(rename.values())]
    return np.log1p(df).mean()


def load_tile_proteins(items_df: pd.DataFrame, proteins: list[str]) -> pd.DataFrame:
    base_to_panel = protein_base_to_panel(proteins)
    sample_id_by_item_id = dict(zip(items_df["id"], items_df["sample_id"]))

    rows = {}
    for item_id, tile_dir in zip(items_df["id"], items_df["tile_dir"]):
        agg = aggregate_tile_proteins(Path(tile_dir), base_to_panel)
        if agg is None:
            continue
        rows[item_id] = agg

    assert rows, "no tiles had a proteins.parquet"
    df = pd.DataFrame.from_dict(rows, orient="index")[proteins]
    df.index.name = "id"
    df["sample_id"] = df.index.map(sample_id_by_item_id)
    return df


def fit_umap(df: pd.DataFrame, proteins: list[str], n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(df[proteins].to_numpy())


def _save_figure(ax: plt.Axes, out_path: Path) -> None:
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def plot_sample_id_umap(embedding: np.ndarray, sample_ids: pd.Series, out_path: Path) -> None:
    samples = sorted(sample_ids.unique())
    palette = dict(zip(samples, sns.color_palette("husl", len(samples))))

    _, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=sample_ids.to_numpy(), hue_order=samples,
        palette=palette, s=8, linewidth=0, alpha=0.7, ax=ax,
    )
    ax.set_title("Tile-level protein UMAP, colored by sample ID")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="sample_id", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2, fontsize="small")
    _save_figure(ax, out_path)


def plot_protein_umap(embedding: np.ndarray, expression: pd.Series, protein: str, out_path: Path) -> None:
    _, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=expression.to_numpy(),
        cmap="viridis", s=8, linewidth=0, alpha=0.7,
    )
    ax.figure.colorbar(points, ax=ax, label="mean log1p intensity")
    ax.set_title(f"Tile-level protein UMAP, colored by {protein}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    _save_figure(ax, out_path)


def compute_umap(cfg: UmapConfig, items_path: Path, out_dir: Path, *, overwrite: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (tile_proteins_df, embedding), using the cache in `out_dir` unless `overwrite`."""
    tile_proteins_path = out_dir / "tile_proteins.parquet"
    embedding_path = out_dir / "embedding.parquet"

    if tile_proteins_path.exists() and embedding_path.exists() and not overwrite:
        logger.info(f"Loading cached tile-level proteins and embedding from {out_dir}")
        df = pd.read_parquet(tile_proteins_path)
        embedding = pd.read_parquet(embedding_path).to_numpy()
        return df, embedding

    items_df = load_items_dataframe(items_path)
    df = load_tile_proteins(items_df, cfg.proteins)
    embedding = fit_umap(df, cfg.proteins, cfg.n_neighbors, cfg.min_dist, cfg.random_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tile_proteins_path)
    pd.DataFrame(embedding, index=df.index, columns=["umap1", "umap2"]).to_parquet(embedding_path)
    return df, embedding


def main(cfg: UmapConfig, *, overwrite: bool = False) -> int:
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    items_path = managed.resolve_items_path(cfg.items_path)
    out_dir = managed.figures_dir / "protein_umap" / items_path.stem

    df, embedding = compute_umap(cfg, items_path, out_dir, overwrite=overwrite)
    plot_sample_id_umap(embedding, df["sample_id"], out_dir / "sample_id.png")
    for protein in cfg.proteins:
        plot_protein_umap(embedding, df[protein], protein, out_dir / f"{items_path.stem}_{protein}.png")
    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(UmapConfig, nested_key="umap")
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.umap, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
