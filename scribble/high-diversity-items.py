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

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# %%
managed_paths = get_managed_paths('beat')

items_dir = managed_paths.output_dir / 'items'

cells = pd.read_json(items_dir / 'cells.json')
cells_stats = pd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/statistics/cells.parquet')

# %%
plt.close('all')
ax = cells_stats['num_unique_cells'].plot.hist(bins=100)
ax.figure.show()

ax = cells_stats.plot.scatter(x='num_unique_transcripts', y='num_unique_cells')
ax.figure.show()

ax = plt.hexbin(x=np.log1p(cells_stats.num_unique_transcripts), y=cells_stats.num_unique_cells)
ax = plt.hexbin(x=cells_stats.num_cells, y=cells_stats.num_unique_cells)
ax.figure.show()

cells_stats.num_cells.plot.hist(bins=100).figure.show()
cells_stats.num_transcripts.plot.hist(bins=100).figure.show()
cells_stats.num_transcripts.sum()

# %%
save_path = managed_paths.output_dir / 'statistics' / 'high_diversity.parquet'
if not save_path.exists():
    from scipy import stats

    incl = ( cells_stats.num_cells > 750 ) & ( cells_stats.num_unique_cells > 30)
    incl.sum()

    incl = incl & ( cells_stats.num_transcripts >= cells_stats.num_transcripts.quantile(q=0.5) )
    incl.sum()

    high_diversity = cells_stats[incl]
    high_diversity.num_transcripts.max()
    high_diversity.num_transcripts.min()
    high_diversity.num_unique_transcripts.min()

    high_diversity = high_diversity.merge(right=cells, left_index=True, right_on='id', how='left')

    cell_type_col = 'Level3_grouped'
    for index, row in tqdm(high_diversity.iterrows()):
        pts = gpd.read_parquet(row.tile_dir + '/cells.parquet')
        high_diversity.loc[index, 'cell_entropy'] = stats.entropy(pts[cell_type_col].value_counts())

        pts = gpd.read_parquet(row.tile_dir + '/transcripts.parquet')
        high_diversity.loc[index, 'transcript_entropy'] = stats.entropy(pts['feature_name'].value_counts())

    high_diversity.to_parquet(save_path)
else:
    high_diversity = pd.read_parquet(save_path)

high_diversity = high_diversity.sort_values(['transcript_entropy', 'cell_entropy'])
high_diversity.transcript_entropy.plot.hist(bins=100).figure.show()
high_diversity.cell_entropy.plot.hist(bins=100).figure.show()

high_diversity[cells.columns].to_json(items_dir / 'high_diversity.json', orient='records')
high_diversity[cells.columns].head(1000).to_json(items_dir / 'low_entropy.json', orient='records')
high_diversity[cells.columns].tail(1000).to_json(items_dir / 'high_entropy.json', orient='records')

# import json
# json.loads((items_dir / 'high_diversity.json').read_text())

# %%
save_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/example-images')
save_dir.mkdir(parents=True, exist_ok=True)
cell_type_col = 'Level3_grouped'
pdat = pd.concat([high_diversity.head(100), high_diversity.tail(100)]).reset_index(drop=True)
for index, row in tqdm(pdat.iterrows()):

    img = torch.load(row.tile_dir + '/tile.pt')
    pts = gpd.read_parquet(row.tile_dir + '/cells.parquet')
    img = img.permute(1,2,0).numpy()

    viz = visualize_points(pts, image=img.copy(), radius=1, color_by_label=True, labels_key='Level3_grouped')

    prefix = 'low' if index < 100 else 'high'
    io.imsave(save_dir / f'{prefix}-{row.id}.png', arr=viz)


