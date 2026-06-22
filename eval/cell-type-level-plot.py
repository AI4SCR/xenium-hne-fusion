# %%
import pandas as pd
from loguru import logger
from xenium_hne_fusion.utils.getters import get_managed_paths

managed_paths = get_managed_paths(name='beat')
results_dir = managed_paths.output_dir / 'results'
result_paths = list(results_dir.rglob('*.parquet'))
logger.info(f'{len(result_paths)} results found')
save_dir = managed_paths.output_dir / 'figures' / 'cell_type_level'
save_dir.mkdir(parents=True, exist_ok=True)

# %%
runs = {
      "expr-token-vit": {"u2l1eisi": 3, "pbi7jz39": 2, "w0y2mwg5": 0, "005s9g5a": 1},
      "expr-token": {"jr7uo9ng": 3, "yeqi6amk": 2, "3pqmw7gh": 1, "m886lfhu": 0},
      "early-fusion": {"owzcohia": 2, "40utmvmw": 0, "fif5wuld": 3, "j5x4suw2": 1},
      # icxsfpqf ua31ay2l cdr1rg1u gnre6bzb
      "vision": {"icxsfpqf": 2, "ua31ay2l": 0, "cdr1rg1u": 3, "gnre6bzb": 1},
  }

runs_to_fold = {
    run:fold for values in runs.values() for run, fold in values.items()
}

runs_to_model = {
    run: key for key, runs in runs.items() for run in runs.keys()
}

container = []
for path in result_paths:
    results = pd.read_parquet(path)
    run_id = path.parent.name
    results['run_id'] = run_id
    results['model'] = results.run_id.map(runs_to_model)
    results['fold'] = results.run_id.map(runs_to_fold)
    container.append(results)

results = pd.concat(container)
results = results.reset_index(drop=True).sort_values(['model', 'fold'])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

metric = 'spearman'
metadata_cols = ['run_id', 'model', 'fold']
cols = results.columns
cols = [c for c in cols if metric in c ]

for col in cols:
    pdat = results[[col] + metadata_cols]
    ax = sns.boxplot(data=pdat, x='model', y=col)
    sns.stripplot(data=pdat, x='model', y=col, hue='fold', ax=ax)

    title = col.replace('test/', '').replace(' ', '_').replace('/', '_')

    ax.set_title(title)
    # ax.figure.show()
    ax.figure.savefig(save_dir / f'{title}.png')
    plt.close('all')

# %%
import json
from pathlib import Path
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL
import torch

items = json.loads((managed_paths.output_dir / 'items' / 'cells.json').read_text())
tile_dir = Path(items[0]["tile_dir"])
cells = pd.read_parquet(tile_dir / "cells.parquet", columns=[DEFAULT_CELL_TYPE_COL])
cell_types = cells[DEFAULT_CELL_TYPE_COL].cat.categories.tolist()
cell_types = ['_'.join(cell_type.split()) for cell_type in cell_types]

for run in runs['expr-token'].keys():
    fold = runs_to_fold[run]
    preds = torch.load(results_dir / run / f'predictions.pt')
    pdat = torch.concat([i['target'] for i in preds])
    pdat = torch.expm1(pdat)
    pdat = pd.DataFrame(pdat.numpy(), columns=cell_types)
    pdat = pdat.sum()
    pdat = pdat.sort_values()
    ax = sns. barplot(data=pdat)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_yscale('log')
    ax.figure.tight_layout()
    ax.figure.show()
    ax.figure.savefig(save_dir / f'proportions-fold={fold}.png')

# %%