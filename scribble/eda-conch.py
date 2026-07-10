# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from skbio.stats.composition import clr
from tqdm import tqdm

from xenium_hne_fusion.train.supervised import CELL_TYPE_COL, CONCH_CLASSES
from xenium_hne_fusion.utils.getters import get_managed_paths

DATASET_NAME = 'beat'
ITEMS_NAME = 'high_diversity.json'
DEBUG_N = 5_000  # cap number of tiles for fast iteration; set to None for the full dataset

CONCH_SCORES_TEMPLATE = (
    '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/results/zeroshot/grid2_256/postxehne/'
    'prompt_atlas_context_0_template_hne_tile_512_stride_256_imsize_224/scores/'
    'raw_scores_postxehne_{sample_id}_atlas_conch_tile_512_stride_256_imsize_224.npy'
)
SAVE_PATH = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/figures/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# classes whose mean score falls below this percentile of the across-class mean-score distribution are flagged as low activation
LOW_ACTIVATION_CLASS_PERCENTILE = 25
# tiles whose max score across classes falls below this percentile of the across-tile max-score distribution are flagged as low score
LOW_SCORE_TILE_PERCENTILE = 10


def load_items(name: str, items_name: str, debug_n: int | None) -> list[dict]:
    managed_paths = get_managed_paths(name)
    items_path = managed_paths.output_dir / 'items' / items_name
    assert items_path.exists(), f'{items_path} does not exist'
    items = json.loads(items_path.read_text())
    np.random.shuffle(items)
    # items = sorted(items, key=lambda item: item['sample_id'])
    return items[:debug_n] if debug_n is not None else items


def load_conch_scores(items: list[dict]) -> pd.DataFrame:
    scores = []
    sample_id, conch_scores = None, None
    for item in tqdm(items, desc='loading conch scores'):
        if sample_id != item['sample_id']:
            sample_id = item['sample_id']
            conch_scores_path = Path(CONCH_SCORES_TEMPLATE.format(sample_id=sample_id))
            assert conch_scores_path.exists(), f'{conch_scores_path} does not exist'
            conch_scores = np.load(conch_scores_path)
        ser = pd.Series(conch_scores[item['tile_id']], index=CONCH_CLASSES, name=item['id'])
        scores.append(ser)
    return pd.concat(scores, axis=1).T


def load_cell_type_counts(items: list[dict]) -> pd.DataFrame:
    counts = []
    for item in tqdm(items, desc='loading cell type counts'):
        cells_path = Path(item['tile_dir']) / 'cells.parquet'
        assert cells_path.exists(), f'{cells_path} does not exist'
        ser = pd.read_parquet(cells_path, columns=[CELL_TYPE_COL])[CELL_TYPE_COL].value_counts()
        ser.name = item['id']
        counts.append(ser)
    return pd.concat(counts, axis=1).T.fillna(0)


def compute_cell_fractions_clr(counts: pd.DataFrame) -> pd.DataFrame:
    frac = (counts + 0.5).div(counts.sum(axis=1), axis=0)
    return pd.DataFrame(clr(frac), index=frac.index, columns=frac.columns)


def compute_correlation(cell_frac_clr: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    assert cell_frac_clr.index.equals(scores.index), 'tile index mismatch between cell fractions and conch scores'
    corr = pd.concat([cell_frac_clr, scores], axis=1).corr()
    return corr.loc[cell_frac_clr.columns, scores.columns]


def plot_correlation_heatmap(corr: pd.DataFrame, save_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    vmax = np.abs(corr.to_numpy()).max()
    fig, ax = plt.subplots(figsize=(0.5 * corr.shape[1] + 3, 0.35 * corr.shape[0] + 2))
    sns.heatmap(
        corr, cmap='vlag', center=0, vmin=-vmax, vmax=vmax, linewidths=0.5, square=True,
        cbar_kws={'label': 'Pearson r'}, ax=ax,
    )
    ax.set_xlabel('conch class')
    ax.set_ylabel('cell type')
    fig.tight_layout()
    out_path = save_path / 'heatmap_cell_type_conch_correlation.png'
    fig.savefig(out_path, dpi=300)
    return out_path


def find_low_activation_classes(scores: pd.DataFrame, percentile: float) -> pd.Series:
    mean_scores = scores.mean(axis=0).sort_values()
    cutoff = np.percentile(mean_scores, percentile)
    return mean_scores[mean_scores < cutoff]


def find_low_score_tiles(scores: pd.DataFrame, percentile: float) -> pd.Series:
    class_cutoffs = scores.apply(lambda col: np.percentile(col, percentile), axis=0)
    is_low_per_class = scores.lt(class_cutoffs, axis=1)
    is_low_tile = is_low_per_class.all(axis=1)
    return scores.max(axis=1)[is_low_tile]


def compute_umap(scores: pd.DataFrame, random_state: int | None = 0) -> pd.DataFrame:
    from umap import UMAP

    embedding = UMAP(random_state=random_state).fit_transform(scores.to_numpy())
    return pd.DataFrame(embedding, index=scores.index, columns=['umap1', 'umap2'])


def plot_umap_by_sample(umap_df: pd.DataFrame, sample_id: pd.Series, save_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = umap_df.join(sample_id)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=df, x='umap1', y='umap2', hue='sample_id', s=8, linewidth=0, alpha=0.7, ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', title='sample_id')
    fig.tight_layout()
    out_path = save_path / 'umap_conch_scores_by_sample.png'
    fig.savefig(out_path, dpi=300)
    return out_path


# %% load data
items = load_items(DATASET_NAME, ITEMS_NAME, DEBUG_N)
scores = load_conch_scores(items)
counts = load_cell_type_counts(items)
scores = scores.loc[counts.index]

# %% correlation heatmap: cell types (clr-transformed fractions) vs. conch scores, per tile
cell_frac_clr = compute_cell_fractions_clr(counts)
corr = compute_correlation(cell_frac_clr, scores)
heatmap_path = plot_correlation_heatmap(corr, SAVE_PATH)
logger.info(f'saved correlation heatmap to {heatmap_path}')

# %% UMAP on conch scores, colored by sample_id
sample_id = pd.Series({item['id']: item['sample_id'] for item in items}, name='sample_id').loc[scores.index]
umap_df = compute_umap(scores)
umap_path = plot_umap_by_sample(umap_df, sample_id, SAVE_PATH)
logger.info(f'saved conch score UMAP to {umap_path}')

# %% flag conch classes with low activation across the dataset
low_activation_classes = find_low_activation_classes(scores, LOW_ACTIVATION_CLASS_PERCENTILE)
logger.info(
    f'{len(low_activation_classes)} / {len(CONCH_CLASSES)} conch classes flagged as low activation '
    f'(mean score below {LOW_ACTIVATION_CLASS_PERCENTILE}th pct):\n{low_activation_classes}'
)

kept_classes = [c for c in CONCH_CLASSES if c not in low_activation_classes.index]
classes_path = SAVE_PATH / 'conch_classes_filtered.json'
classes_path.write_text(json.dumps(kept_classes, indent=2))
logger.info(f'saved {len(kept_classes)} kept conch classes to {classes_path}')

# %% flag tiles with low score across all conch classes
low_score_tiles = find_low_score_tiles(scores, LOW_SCORE_TILE_PERCENTILE)
logger.info(
    f'{len(low_score_tiles)} / {len(scores)} tiles flagged as low score '
    f'(max score across classes below {LOW_SCORE_TILE_PERCENTILE}th pct)'
)

kept_item_ids = set(scores.index) - set(low_score_tiles.index)
kept_items = [item for item in items if item['id'] in kept_item_ids]
items_path = SAVE_PATH / 'items_filtered.json'
items_path.write_text(json.dumps(kept_items, indent=2))
logger.info(f'saved {len(kept_items)} / {len(items)} kept items to {items_path}')

# %%
