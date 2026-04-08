from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from loguru import logger
from scipy import sparse

from xenium_hne_fusion.metadata import load_items_dataframe


def load_fit_items(items_path: Path, split_metadata_path: Path) -> pd.DataFrame:
    items = load_items_dataframe(items_path)
    split_metadata = pd.read_parquet(split_metadata_path)
    fit_ids = set(split_metadata.index[split_metadata['split'] == 'fit'])
    fit_items = items[items['id'].isin(fit_ids)].copy()
    assert len(fit_items) > 0, f'No fit items found for {items_path} and {split_metadata_path}'
    return fit_items


def get_common_genes(fit_items: pd.DataFrame) -> list[str]:
    sample_ids = fit_items['sample_id'].drop_duplicates().tolist()
    sample_gene_orders = {sample_id: _load_sample_gene_order(fit_items, sample_id) for sample_id in sample_ids}

    common_genes = set(sample_gene_orders[sample_ids[0]])
    for sample_id in sample_ids[1:]:
        common_genes &= set(sample_gene_orders[sample_id])

    canonical_order = [gene for gene in sample_gene_orders[sample_ids[0]] if gene in common_genes]
    assert canonical_order, 'No common genes found across selected fit samples'
    return canonical_order


def build_tile_level_matrix(
    fit_items: pd.DataFrame,
    genes: list[str],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    rows = []
    obs_rows = []

    for item in fit_items.itertuples(index=False):
        counts = load_tile_gene_counts(Path(item.tile_dir) / 'transcripts.parquet', genes)
        summed = counts.to_numpy(dtype=np.float32, copy=False)
        rows.append(sparse.csr_matrix(summed[np.newaxis, :]))
        obs_rows.append({'id': item.id, 'sample_id': item.sample_id, 'tile_id': item.tile_id})

    matrix = sparse.vstack(rows, format='csr')
    obs = pd.DataFrame(obs_rows).set_index('id', drop=True)
    return matrix, obs


def build_hvg_anndata(
    fit_items: pd.DataFrame,
) -> AnnData:
    genes = get_common_genes(fit_items)
    matrix, obs = build_tile_level_matrix(fit_items, genes)
    var = pd.DataFrame(index=pd.Index(genes, name='gene'))
    return AnnData(X=matrix, obs=obs, var=var)


def select_highly_variable_genes(adata: AnnData, *, n_top_genes: int, flavor: str) -> list[str]:
    import scanpy as sc

    assert n_top_genes > 0, 'n_top_genes must be positive'
    assert n_top_genes <= adata.n_vars, f'n_top_genes={n_top_genes} exceeds available genes={adata.n_vars}'

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        inplace=True,
    )
    hvg_mask = adata.var['highly_variable'].fillna(False).astype(bool)
    return adata.var_names[hvg_mask].tolist()


def save_hvg_panel(output_path: Path, genes: list[str], hvg_genes: list[str], overwrite: bool = False) -> Path:
    if output_path.exists():
        assert overwrite, f'Panel already exists: {output_path}'

    hvg_set = set(hvg_genes)
    target_panel = [gene for gene in genes if gene in hvg_set]
    source_panel = [gene for gene in genes if gene not in hvg_set]

    assert target_panel, 'No HVGs selected'
    assert set(source_panel).isdisjoint(set(target_panel))
    assert len(source_panel) + len(target_panel) == len(genes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            {'source_panel': source_panel, 'target_panel': target_panel},
            sort_keys=False,
        )
    )
    logger.info(f'Saved HVG panel → {output_path}')
    return output_path


def create_panel(
    *,
    items_path: Path,
    split_metadata_path: Path,
    output_path: Path,
    n_top_genes: int,
    flavor: str = 'seurat_v3',
    overwrite: bool = False,
) -> Path:
    fit_items = load_fit_items(items_path, split_metadata_path)
    adata = build_hvg_anndata(fit_items)
    hvg_genes = select_highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    return save_hvg_panel(output_path, adata.var_names.tolist(), hvg_genes, overwrite=overwrite)


def _load_sample_gene_order(fit_items: pd.DataFrame, sample_id: str) -> list[str]:
    sample_items = fit_items[fit_items['sample_id'] == sample_id]
    assert len(sample_items) > 0, f'No items found for sample_id={sample_id}'

    transcripts_path = Path(sample_items.iloc[0]['tile_dir']) / 'transcripts.parquet'
    return load_transcript_gene_categories(transcripts_path)


def load_transcript_gene_categories(transcripts_path: Path) -> list[str]:
    transcripts = pd.read_parquet(transcripts_path, columns=['feature_name'])
    feature_name = transcripts['feature_name']
    assert isinstance(feature_name.dtype, pd.CategoricalDtype), (
        f'Expected categorical feature_name in {transcripts_path}, got {feature_name.dtype}'
    )
    return feature_name.cat.categories.tolist()


def load_tile_gene_counts(transcripts_path: Path, genes: list[str]) -> pd.Series:
    transcripts = pd.read_parquet(transcripts_path, columns=['feature_name'])
    feature_name = transcripts['feature_name']
    assert isinstance(feature_name.dtype, pd.CategoricalDtype), (
        f'Expected categorical feature_name in {transcripts_path}, got {feature_name.dtype}'
    )
    counts = feature_name.value_counts(sort=False)
    return counts.reindex(genes, fill_value=0)
