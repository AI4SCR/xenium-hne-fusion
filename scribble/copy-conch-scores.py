# %%
import json
from pathlib import Path
from xenium_hne_fusion.train.supervised import CONCH_CLASSES
from xenium_hne_fusion.utils.getters import get_managed_paths

save_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/figures/')

classes = CONCH_CLASSES

managed_paths = get_managed_paths('beat')
items_name = 'high_diversity.json'
items_path = managed_paths.output_dir / 'items' / items_name
assert items_path.exists()
items = json.loads(items_path.read_text())
items = sorted(items, key=lambda item: item['sample_id'])

import numpy as np
import pandas as pd
from tqdm import tqdm

sample_ids = set([i['sample_id'] for i in items])
template = '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/zeroshot/grid2_256/postxehne/prompt_atlas_context_0_template_hne_tile_512_stride_256_imsize_224/scores/raw_scores_postxehne_{sample_id}_atlas_conch_tile_512_stride_256_imsize_224.npy'
# scores = []
# for sample_id in tqdm(sample_ids):
#     conch_scores_path = Path(template.format(sample_id=sample_id))
#     assert conch_scores_path.exists()
#     conch_scores = pd.DataFrame(np.load(conch_scores_path), columns=CONCH_CLASSES)
#     conch_scores['sample_id'] = sample_id
#     scores.append(conch_scores)
#
# scores = pd.concat(scores)

# %%
scores = []
prev_sample_id, conch_scores = None, None
for item in items:
    sample_id = item['sample_id']
    tile_id = item['tile_id']

    if prev_sample_id is None or prev_sample_id != sample_id:
        prev_sample_id = sample_id
        conch_scores_path = Path(template.format(sample_id=item['sample_id']))
        assert conch_scores_path.exists()
        conch_scores = np.load(conch_scores_path)
        # tmp = np.load(f'/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/zeroshot/grid2_256/postxehne/prompt_atlas_context_0_template_hne_tile_512_stride_256_imsize_224/scores/classes_postxehne_{sample_id}_atlas_conch_tile_512_stride_256_imsize_224.npy')
        # tmp[tile_id]

    ser = pd.Series(conch_scores[tile_id], index=CONCH_CLASSES, name=item['id'])
    scores.append(ser)
    conch_class = conch_scores[tile_id].argsort()[::-1]
    item['conch_class'] = conch_class[0].item()
    # item['conch_scores']

json.dump(items, (save_path / 'items_with_conch_labels.json').open('w'))
# items = json.load((save_path / 'items_with_conch_labels.json').open('r'))
scores = pd.concat(scores, axis=1).T

# %%
import pandas as pd
from tqdm import tqdm

if False and (save_path / 'cells.parquet').exists():
    cells = pd.read_parquet(save_path / 'cells.parquet', engine='pyarrow')
else:
    expr = []
    cells = []
    labels = []
    for i, item in tqdm(enumerate(items), total=len(items)):

        # labels.append(item['conch_class'])

        # expr_token_path = Path(item['tile_dir']) / 'expr-kernel_size=16.parquet'
        # expr_token = pd.read_parquet(expr_token_path)
        # expr.append(expr_token.sum())

        cells_path = Path(item['tile_dir']) / 'cells.parquet'
        ser = pd.read_parquet(cells_path).Level3_grouped.value_counts().T
        ser.name = item['id']
        cells.append(ser)

        if i == 50:
            break

    # index = pd.Index(labels, name='label')
    # index.value_counts()

    # expr = pd.concat(expr, axis=1).T
    # expr.index = index
    # expr.to_parquet(save_path / 'expr.parquet')

    cells = pd.concat(cells, axis=1).T
    # cells.index = index
    cells.to_parquet(save_path / 'cells.parquet')

# %%
scores = scores.loc[cells.index]


# %%
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.stats.composition import clr

label_names = {i: v for i, v in enumerate(classes)}

frac = (cells + 0.5).div(cells.sum(axis=1), axis=0)
frac = frac.fillna(0)
frac = pd.DataFrame(clr(frac), index=frac.index, columns=frac.columns)
mean_by_class = frac.groupby("label").mean()
heatmap_values = mean_by_class.apply(zscore, axis=0)
heatmap_values.index = heatmap_values.index.map(label_names)
vmax = np.abs(heatmap_values.to_numpy()).max()

fig, ax = plt.subplots(figsize=(10, 0.3 * heatmap_values.shape[0]))
ax = sns.heatmap(
    heatmap_values,
    cmap="vlag",  # or "coolwarm", "vlag", "RdBu_r"
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Z-score"},
    ax=ax
)

ax.figure.tight_layout()
ax.figure.show()
ax.figure.savefig(save_path / 'heatmap_cell_fractions_by_conch_class.png', dpi=300)

# %%
