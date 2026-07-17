from pathlib import Path
from itertools import combinations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from xenium_hne_fusion.utils.getters import ManagedPaths

paths = ManagedPaths(
    data_dir=Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v4'),
    name='owkin',
)

save_dir = paths.output_dir / 'figures'
save_dir.mkdir(parents=True, exist_ok=True)


def plot_overlap_heatmap(universes: dict[str, set[str]], title: str, save_path: Path) -> None:
    records = []
    for (s1, v1), (s2, v2) in combinations(universes.items(), 2):
        records.append({'sample_1': s1, 'sample_2': s2, 'intersection': len(v1 & v2)})

    df = pd.DataFrame.from_records(records)
    df = df.pivot(index='sample_1', columns='sample_2', values='intersection')

    _, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
    ax.set_title(title)
    ax.figure.tight_layout()
    ax.figure.savefig(save_path)
    plt.close(ax.figure)


# %% gene overlap

gene_universes = {p.parent.name: set(p.read_text().split())
                  for p in sorted(paths.processed_dir.glob('*/feature_universe.txt'))}

beat_panel = yaml.load(Path('panels/beat/default.yaml').open(), Loader=yaml.SafeLoader)
gene_universes['beat'] = set(beat_panel['source_panel']) | set(beat_panel['target_panel'])

plot_overlap_heatmap(gene_universes, 'owkin gene panel overlap', save_dir / 'gene-overlap.png')

# %% protein overlap

protein_universes = {
    sample_dir.name: set(pd.read_parquet(sample_dir / 'proteins.parquet').columns) - {'geometry'}
    for sample_dir in sorted(paths.structured_dir.iterdir())
    if (sample_dir / 'proteins.parquet').exists()
}

plot_overlap_heatmap(protein_universes, 'owkin protein panel overlap', save_dir / 'protein-overlap.png')

# %%

panel = {'source_panel': sorted(gene_universes['CH_C_518a_x2'] & gene_universes['CH_D_529a_x2'] & gene_universes['beat'])}
len(panel['source_panel'])
save_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/panels/owkin/owkin-beat.yaml')
save_path.parent.mkdir(parents=True, exist_ok=True)
yaml.dump(panel, save_path.open('w'))
