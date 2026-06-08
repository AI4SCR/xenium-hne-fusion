# %%
from dotenv import load_dotenv
assert load_dotenv(), 'no .env found'

from pathlib import Path
from xenium_hne_fusion.utils.getters import get_managed_paths
import pandas as pd

import torch
from skimage import io
from ai4bmr_learn.plotting.xenium import visualize_points
import geopandas as gpd

# %%
managed_paths = get_managed_paths('beat')

tiles = '512_256'
managed_paths.processed_dir

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
cells_stats = pd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/statistics/cells.parquet')
ax = cells_stats['num_unique_cells'].plot.hist(bins=100)
ax.figure.show()

ax = cells_stats.plot.scatter(x='num_unique_transcripts', y='num_unique_cells')
ax.figure.show()

ax = plt.hexbin(x=np.log1p(cells_stats.num_unique_transcripts), y=cells_stats.num_unique_cells)
ax = plt.hexbin(x=cells_stats.num_cells, y=cells_stats.num_unique_cells)
ax.figure.show()

ax = cells_stats.num_cells.plot.hist(bins=100)
ax.figure.show()

incl = ( cells_stats.num_cells > 500 ) & ( cells_stats.num_unique_cells > 30)
high_diversity = cells_stats[incl]

cells = pd.read_json(managed_paths.output_dir / 'items' / 'cells.json')
high_diversity = high_diversity.merge(right=cells, left_index=True, right_on='id', how='left')

# %%
from scipy import stats
from tqdm import tqdm
cell_type_col = 'Level3_grouped'
for index, row in tqdm(high_diversity.iterrows()):
    pts = gpd.read_parquet(row.tile_dir + '/cells.parquet')
    high_diversity.loc[index, 'entropy'] = stats.entropy(pts[cell_type_col].value_counts())

low_entropy = high_diversity.sort_values('entropy').head(100)
high_entropy = high_diversity.sort_values('entropy').tail(100)

# %%
save_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/example-images')
save_dir.mkdir(parents=True, exist_ok=True)
cell_type_col = 'Level3_grouped'
pdat = pd.concat([low_entropy, high_entropy]).reset_index(drop=True)
for index, row in tqdm(pdat.iterrows()):

    img = torch.load(row.tile_dir + '/tile.pt')
    pts = gpd.read_parquet(row.tile_dir + '/cells.parquet')
    img = img.permute(1,2,0).numpy()
    entropy = stats.entropy(pts[cell_type_col].value_counts())
    viz = visualize_points(pts, image=img.copy(), radius=1, color_by_label=True, labels_key='Level3_grouped')

    prefix = 'low' if index < 100 else 'high'
    io.imsave(save_dir / f'{prefix}-{row.id}.png', arr=viz)
