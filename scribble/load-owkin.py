import numpy as np
import torch
import lazyslide as zs

from pathlib import Path
import geopandas as gpd
import scanpy as sc


df = gpd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/owkin_spatial/data/crc_gbm_dlbcl_pilot/xenium_multimodal/results/matched_samples_cells_aligned/CH_C_518a_x2/centroids/centroid_cropped.parquet')

df.columns

# %% geometry is transformed, # 1.11.5
xenium_sub_dir="normalised_results/outs"

sample_id = 'CH_C_518a_x2'
data_path = Path("/work/PRTNR/CHUV/DIR/rgottar1/owkin_spatial/data/crc_gbm_dlbcl_pilot/xenium_multimodal/results/")
xenium_path = data_path / "matched_samples" / sample_id / "xenium"
h5_dir_path = xenium_path / xenium_sub_dir / "cell_feature_matrix"
h5_dir_path.exists()
adata = sc.read_10x_mtx(h5_dir_path, gex_only=False)

# %%

adata.var.feature_types.value_counts()
adata.X[:5, :5]

adata.var
protein_cols = adata.var.feature_types == 'Protein Expression'
adata.X[:50, protein_cols.values].toarray()
adata.X[:, protein_cols.values].max()
cell_ids = set(adata.obs.index)

set(df.cell_id) == cell_ids

len(cell_ids - set(df.cell_id))
set(df.cell_id) - cell_ids
df.cell_id.shape
