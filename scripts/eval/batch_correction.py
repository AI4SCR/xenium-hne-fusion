"""
Compare batch-effect correction methods on cell-level protein expression.

Loads whole-sample protein intensities (`01_structured/<name>/<sample_id>/proteins.parquet`)
for every sample in an item set (e.g. `c_cells`), treating `sample_id` as the batch variable
(protein_histograms.py showed strong per-sample shifts in marker intensity within c_cells,
which is all one cancer type -- so sample_id is the only technical batch here, not confounded
with cancer type). log1p + z-scales the panel, computes a baseline PCA embedding, then runs
three correction methods on top. Each method's corrected embedding is scored with 3 batch-
mixing metrics, and everything (raw + corrected embeddings + metrics) is written to one
AnnData, saved as zarr so the corrected embeddings can be reused by downstream tasks (e.g.
clustering, protein_umap.py) without recomputing them.

## Method selection (literature-grounded)

Candidates considered: Harmony, ComBat, Scanorama, BBKNN, scVI.
- Harmony: operates directly on continuous low-dimensional PCA embeddings (no count-data
  assumption), and is the method most consistently recommended across integration benchmarks,
  including spatial-proteomics/CODEX-specific evaluations (Nature Methods 2021 scIB benchmark;
  BioBatchNet 2025 IMC benchmark).
- ComBat: classical empirical-Bayes linear correction, the standard baseline for continuous
  marker-intensity data (mass-spec proteomics, CyTOF) rather than sparse RNA counts -- built
  for exactly this data type, and already available via `scanpy.pp.combat`.
- Scanorama: MNN-based, algorithmically distinct from Harmony (soft clustering) and ComBat
  (linear model); scIB found it integrates across strong batch effects while retaining
  biological variation.
- Not selected: BBKNN only corrects the neighbor graph, not the expression/embedding matrix,
  so it can't be scored with the same embedding-based metrics or reused downstream, and scIB
  found it collapses fine-grained biological variation. scVI assumes (zero-inflated)
  negative-binomial count data -- a poor fit for continuous, already-compensated protein
  intensities; the BioBatchNet IMC benchmark had to hack its likelihood to Gaussian to use it
  at all.

## Metrics
- batch_asw: 1 - |silhouette score| on batch labels in the embedding (near 1 = well mixed).
- batch_knn_entropy: mean Shannon entropy of batch composition within each cell's k nearest
  neighbors, normalized by log(n_batches) (near 1 = well mixed).
- batch_pcr: mean R^2 of a linear regression of each embedding dimension on one-hot batch
  labels (near 0 = batch effect removed).

Usage:
    uv run python scripts/eval/batch_correction.py \\
        --batch_correction.name owkin \\
        --batch_correction.data_dir $DATA_DIR \\
        --batch_correction.items_path c_cells.json \\
        --batch_correction.debug true
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from xenium_hne_fusion.artifacts.items import load_items_dataframe
from xenium_hne_fusion.targets import PROTEIN_PANEL, load_sample_proteins
from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class BatchCorrectionConfig:
    name: str = None
    data_dir: Path = None
    items_path: Path = Path("c_cells.json")
    proteins: list[str] = field(default_factory=lambda: list(PROTEIN_PANEL))
    n_pca_comps: int = 15
    n_neighbors: int = 30
    random_state: int = 0
    debug: bool = False
    debug_cells_per_batch: int = 500


def load_cell_proteins(structured_dir: Path, sample_ids: list[str], proteins: list[str]) -> pd.DataFrame:
    frames = []
    for sample_id in sample_ids:
        df = load_sample_proteins(structured_dir, sample_id, proteins)
        df["sample_id"] = sample_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def subsample_per_batch(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n), random_state=random_state) for _, g in df.groupby("sample_id")]
    return pd.concat(parts, ignore_index=True)


def build_adata(cells: pd.DataFrame, proteins: list[str]) -> AnnData:
    adata = AnnData(X=np.log1p(cells[proteins].to_numpy(dtype=np.float32)))
    adata.var_names = proteins
    adata.obs["sample_id"] = pd.Categorical(cells["sample_id"].to_numpy())
    return adata


def batch_asw(embedding: np.ndarray, batch: pd.Series) -> float:
    return 1 - abs(silhouette_score(embedding, batch))


def batch_knn_entropy(embedding: np.ndarray, batch: pd.Series, n_neighbors: int) -> float:
    batch_codes = batch.cat.codes.to_numpy()
    n_batches = len(batch.cat.categories)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    _, neighbor_idx = nn.kneighbors(embedding)

    counts = np.stack([np.bincount(batch_codes[row], minlength=n_batches) for row in neighbor_idx])
    p = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(p * np.log(p, out=np.zeros_like(p), where=p > 0), axis=1)
    return float(entropy.mean() / np.log(n_batches))


def batch_pcr(embedding: np.ndarray, batch: pd.Series) -> float:
    dummies = pd.get_dummies(batch).to_numpy(dtype=np.float32)
    return float(LinearRegression().fit(dummies, embedding).score(dummies, embedding))


def evaluate_embedding(embedding: np.ndarray, batch: pd.Series, n_neighbors: int) -> dict:
    return {
        "batch_asw": batch_asw(embedding, batch),
        "batch_knn_entropy": batch_knn_entropy(embedding, batch, n_neighbors),
        "batch_pcr": batch_pcr(embedding, batch),
    }


def run_harmony(embedding: np.ndarray, batch: pd.Series, random_state: int) -> np.ndarray:
    import harmonypy

    out = harmonypy.run_harmony(embedding.astype(np.float64), pd.DataFrame({"batch": batch}), "batch", random_state=random_state)
    # harmonypy>=2.0.0 (pinned in pyproject.toml) returns Z_corr as (n_cells, n_pcs); the scanpy
    # wrapper still assumes the pre-2.0 (n_pcs, n_cells) convention and silently mis-transposes it.
    z = out.Z_corr
    assert z.shape == embedding.shape, f"unexpected harmonypy Z_corr shape {z.shape}, expected {embedding.shape}"
    return z


def main(cfg: BatchCorrectionConfig) -> int:
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    items_path = managed.resolve_items_path(cfg.items_path)

    items_df = load_items_dataframe(items_path)
    sample_ids = sorted(items_df["sample_id"].unique())
    logger.info(f"Loading proteins for {len(sample_ids)} samples: {sample_ids}")

    cells = load_cell_proteins(managed.structured_dir, sample_ids, cfg.proteins)
    if cfg.debug:
        cells = subsample_per_batch(cells, cfg.debug_cells_per_batch, cfg.random_state)
    logger.info(f"Loaded {len(cells)} cells")

    adata = build_adata(cells, cfg.proteins)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=cfg.n_pca_comps, random_state=cfg.random_state)

    logger.info("Running Harmony...")
    adata.obsm["X_pca_harmony"] = run_harmony(adata.obsm["X_pca"], adata.obs["sample_id"], cfg.random_state)

    logger.info("Running ComBat...")
    combat_x = sc.pp.combat(adata, key="sample_id", inplace=False)
    adata.layers["combat"] = combat_x
    adata.obsm["X_pca_combat"] = PCA(n_components=cfg.n_pca_comps, random_state=cfg.random_state).fit_transform(combat_x)

    logger.info("Running Scanorama...")
    sc.external.pp.scanorama_integrate(adata, "sample_id")

    embeddings = {
        "uncorrected": "X_pca",
        "harmony": "X_pca_harmony",
        "combat": "X_pca_combat",
        "scanorama": "X_scanorama",
    }
    metrics = {
        method: evaluate_embedding(adata.obsm[obsm_key], adata.obs["sample_id"], cfg.n_neighbors)
        for method, obsm_key in embeddings.items()
    }

    metrics_df = pd.DataFrame(metrics).T
    logger.info(f"Batch-correction metrics:\n{metrics_df}")
    adata.uns["batch_correction_metrics"] = metrics_df.to_dict(orient="index")

    out_dir = managed.anndata_dir / "batch_correction"
    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write_zarr(out_dir / f"{items_path.stem}.zarr")
    metrics_df.to_csv(out_dir / f"{items_path.stem}_metrics.csv")

    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(BatchCorrectionConfig, nested_key="batch_correction")
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.batch_correction)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
