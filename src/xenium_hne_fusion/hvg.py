from pathlib import Path

import pandas as pd
import yaml
from anndata import AnnData
from loguru import logger
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenium_hne_fusion.datasets.tiles import TileDataset


def get_common_genes(split_metadata: pd.DataFrame, processed_dir: Path) -> list[str]:
    assert len(split_metadata) > 0, 'No split metadata provided'
    assert 'split' in split_metadata.columns, 'Missing split column'
    assert 'sample_id' in split_metadata.columns, 'Missing sample_id column'
    fit_metadata = split_metadata.loc[split_metadata['split'] == 'fit']
    assert len(fit_metadata) > 0, 'No fit items provided'
    gene_orders = [
        load_feature_universe(processed_dir / sample_id / 'feature_universe.txt')
        for sample_id in fit_metadata['sample_id'].unique()
    ]

    common_genes = set(gene_orders[0])
    for gene_order in gene_orders[1:]:
        common_genes &= set(gene_order)

    canonical_order = [gene for gene in gene_orders[0] if gene in common_genes]
    assert canonical_order, 'No common genes found across selected fit samples'
    return canonical_order


def build_hvg_anndata_from_split(
    *,
    items_path: Path,
    split_metadata_path: Path,
    genes: list[str],
    batch_size: int = 256,
    num_workers: int = 10,
) -> AnnData:
    ds = TileDataset(
        target='expression',
        source_panel=None,
        target_panel=genes,
        include_image=False,
        include_expr=False,
        expr_pool='tile',
        items_path=items_path,
        metadata_path=split_metadata_path,
        split='fit',
        id_key='id',
    )
    ds.setup()
    assert len(ds) > 0, f'No fit items found for {items_path} and {split_metadata_path}'

    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    rows = []
    obs_rows = []
    for batch in tqdm(dl, desc='HVG'):
        target = batch['target']
        rows.append(sparse.csr_matrix(target.numpy()))
        obs_rows.extend(
            {
                'id': item_id,
                'sample_id': sample_id,
                'tile_id': int(tile_id),
            }
            for item_id, sample_id, tile_id in zip(batch['id'], batch['sample_id'], batch['tile_id'], strict=True)
        )

    matrix = sparse.vstack(rows, format='csr')
    obs = pd.DataFrame(obs_rows).set_index('id', drop=True)
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
    processed_dir: Path,
    output_path: Path,
    n_top_genes: int,
    flavor: str = 'seurat_v3',
    batch_size: int = 256,
    num_workers: int = 10,
    overwrite: bool = False,
) -> Path:
    split_metadata = pd.read_parquet(split_metadata_path)
    common_genes = get_common_genes(split_metadata, processed_dir)
    num_common_genes = len(common_genes)
    assert 'sample_id' in split_metadata.columns, 'Missing sample_id column'
    num_fit_samples = split_metadata.loc[split_metadata['split'] == 'fit', 'sample_id'].nunique()
    logger.info(f'Found {num_common_genes} common genes across {num_fit_samples} fit samples')
    assert num_common_genes >= n_top_genes, (
        f'n_top_genes={n_top_genes} exceeds common genes={num_common_genes}'
    )

    adata = build_hvg_anndata_from_split(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        genes=common_genes,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    hvg_genes = select_highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    return save_hvg_panel(output_path, adata.var_names.tolist(), hvg_genes, overwrite=overwrite)


def load_transcript_gene_categories(transcripts_path: Path) -> list[str]:
    transcripts = pd.read_parquet(transcripts_path, columns=['feature_name'])
    feature_name = transcripts['feature_name']
    assert isinstance(feature_name.dtype, pd.CategoricalDtype), (
        f'Expected categorical feature_name in {transcripts_path}, got {feature_name.dtype}'
    )
    return feature_name.cat.categories.tolist()


def load_feature_universe(feature_universe_path: Path) -> list[str]:
    assert feature_universe_path.exists(), f'Missing feature universe: {feature_universe_path}'
    genes = feature_universe_path.read_text().splitlines()
    assert genes, f'Empty feature universe: {feature_universe_path}'
    return genes
