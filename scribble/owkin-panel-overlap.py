from pathlib import Path

data_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/02_processed/owkin')
universes = {p.parent.name: set(p.read_text().split())
             for p in sorted(data_dir.glob('*/feature_universe.txt'))}
len(universes)
universes.keys()

import yaml
beat_panel = yaml.load(Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/panels/beat/default.yaml').open(), Loader=yaml.SafeLoader)
universes['beat'] = set(beat_panel['source_panel']) | set(beat_panel['target_panel'])

from itertools import combinations
records = []
for (s1, v1), (s2, v2) in combinations(universes.items(), 2):
    records.append({'sample_1': s1, 'sample_2': s2, 'intersection': len(v1.intersection(v2))})

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_records(records)
df = df.pivot(index='sample_1', columns='sample_2', values='intersection')

_, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
ax.figure.tight_layout()
ax.figure.show()
save_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/owkin/figures')
ax.figure.savefig(save_path)

# %%

panel = {'source_panel': sorted(universes['CH_C_518a_x2'] & universes['CH_D_529a_x2'] & universes['beat'])}
len(panel['source_panel'])
save_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/panels/owkin/owkin-beat.yaml')
save_path.parent.mkdir(parents=True, exist_ok=True)
yaml.dump(panel, save_path.open('w'))
