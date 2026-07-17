"""EDA: per-protein value distributions across randomly sampled CH_C tiles.

Motivation: proteins.parquet stores per-cell protein intensities as integers, so they
may behave like counts (non-negative, possibly overdispersed) rather than continuous
intensities. This informs whether arcsinh (used for cytometry intensities that can go
negative) is the right transform, or whether a count-oriented transform (e.g. log1p)
fits better.
"""
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xenium_hne_fusion.targets import PROTEIN_PANEL
from xenium_hne_fusion.utils.getters import ManagedPaths

paths = ManagedPaths(
    data_dir=Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v4'),
    name='owkin',
)

save_dir = paths.figures_dir
save_dir.mkdir(parents=True, exist_ok=True)

NUM_TILES = 1000
SEED = 0

# %% sample tiles from CH_C samples

tile_paths = list(paths.processed_dir.glob('CH_C_*/512_256/*/proteins.parquet'))
assert tile_paths, f'No proteins.parquet found under {paths.processed_dir}/CH_C_*/512_256/*'

random.seed(SEED)
sampled_paths = random.sample(tile_paths, min(NUM_TILES, len(tile_paths)))
print(f'Sampled {len(sampled_paths)} / {len(tile_paths)} CH_C tiles')

# %% load per-cell protein values across sampled tiles

cells = pd.concat(
    (pd.read_parquet(p, columns=PROTEIN_PANEL) for p in sampled_paths),
    ignore_index=True,
)
print(f'Loaded {len(cells)} cells')
print(cells.describe().T)

# %% plot per-protein distributions, raw and under log1p/arcsinh


def plot_distributions(values: pd.DataFrame, title: str, save_path: Path) -> None:
    ncols = 5
    nrows = math.ceil(len(PROTEIN_PANEL) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, protein in zip(axes.flat, PROTEIN_PANEL, strict=False):
        ax.hist(values[protein], bins=50)
        ax.set_title(protein)
        ax.set_yscale('log')
    for ax in axes.flat[len(PROTEIN_PANEL):]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved -> {save_path}')


transforms = {'raw': lambda x: x, 'log1p': np.log1p, 'arcsinh': np.arcsinh}
for name, transform in transforms.items():
    plot_distributions(transform(cells), name, save_dir / f'protein-distributions-{name}.png')
