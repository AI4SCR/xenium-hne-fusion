# %%
from dotenv import load_dotenv

assert load_dotenv(override=True)

from pathlib import Path
from loguru import logger
import pandas as pd
from xenium_hne_fusion.utils.getters import get_managed_paths
import pyarrow.parquet as pq
from tqdm import tqdm

from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

import seaborn as sns

# %%
dataset_name = 'beat'
paths = get_managed_paths(dataset_name)
output_dir = paths.output_dir

items_path = output_dir / 'items' / 'cells.json'
assert items_path.exists(), f'items_path not found: {items_path}'
items = pd.read_json(items_path)
sample_ids = set(items.sample_id)

paths = get_managed_paths('beat')
save_dir = paths.output_dir / f'baselines' / 'expression'

# %%
master = pd.read_csv('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/18032026_metadata_beat_meso.csv')
include = master.paired_id.isin(sample_ids) & (master.tech_id == 'XE')
sid_to_pid = master.loc[include].set_index('paired_id')['clinical_pat_id'].to_dict()
assert set(sid_to_pid) == sample_ids, f'master coverage mismatch: {sample_ids - set(sid_to_pid)}'

# %%
for i, sid in tqdm(enumerate(sample_ids)):
    logger.info(f'[{i+1}/{len(sample_ids)}] Processing {sid}')

    save_path = save_dir / f'{sid}.parquet'
    if save_path.exists():
        continue

    save_path.parent.mkdir(parents=True, exist_ok=True)

    transcripts_path = paths.structured_dir / sid / 'transcripts.parquet'
    pf = pq.ParquetFile(transcripts_path)
    assert transcripts_path.exists()

    chunked = []
    for batch in pf.iter_batches(batch_size=100_000, columns=['feature_name']):
        chunk = batch.to_pandas()
        chunked.append(chunk.value_counts('feature_name').to_frame())
    counts = pd.concat(chunked, axis=1).sum(axis=1).convert_dtypes()
    counts = counts.to_frame().rename(columns={0: 'count'}).sort_index()
    counts.to_parquet(save_path)

# %%
import json
import numpy as np
clinical = pd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/metadata_b.parquet')
clinical = clinical.drop_duplicates()
clinical = clinical.set_index('clinical_pat_id')
assert clinical.index.is_unique

data = []
for p in save_dir.glob('*.parquet'):
    df = pd.read_parquet(p).rename(columns={'count': p.stem})
    data.append(df)
data = pd.concat(data, axis=1).T
data.index = data.index.map(sid_to_pid)
data.index.name = 'clinical_pat_id'
# NOTE: aggregation across different samples of the same patient
data = data.groupby('clinical_pat_id').mean()
assert data.index.is_unique

clinical = clinical.loc[data.index]

clinical['7b'].value_counts()
clinical['7b'].isna().sum()

# %%
import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self, num: int, num_hidden: int = 0):
        super().__init__()

        layers = []
        for _ in range(num_hidden):
            layers.extend([nn.Linear(num, num), nn.ReLU()])
        layers.append(nn.Linear(num, 1))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def fit(model: nn.Module, x: torch.tensor, time: torch.tensor, event: torch.tensor, num_epochs: int = 100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    time.to(device)
    event.to(device)
    x.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    monitor = []
    cindex = ConcordanceIndex()
    for epoch in range(100):
        optimizer.zero_grad()

        estimate = model(x)
        loss = cox.neg_partial_log_likelihood(estimate, event=event, time=time)

        loss.backward()
        optimizer.step()

        cindex_score = cindex(estimate, event, time).item()
        monitor.append({
            'epoch': epoch,
            'loss': loss.item(),
            'cindex': cindex_score
        })
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | cindex: {cindex_score}')

    return monitor

# %%
from sklearn.preprocessing import StandardScaler
splits = json.load(Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/metadata_b_splits.json').open())

data = np.log1p(data)

# time = time_train
# event = event_train
# x = x_train

pfs_time = '12'
pfs_event = '13'

os_time = '52'
os_event = '53'

targets = [('pfs', pfs_time, pfs_event), ('os', os_time, os_event)]

for target in targets:
    target_name, time_col, event_col = target
    scores = []
    for num_epochs in [50, 100, 250]:
        for num_hidden in [0, 1, 2]:
            for split in splits:
                model_name = f'epochs={num_epochs}-hidden={num_hidden}'
                scaler = StandardScaler()

                train_ids = split['train_ids']
                test_ids = split['test_ids']

                time_train = torch.tensor(clinical.loc[train_ids, time_col].values.astype(float)).float()
                event_train = torch.tensor(clinical.loc[train_ids, event_col].values.astype(int)).bool()
                x_train = data.loc[train_ids].values.astype(float)
                x_train = scaler.fit_transform(x_train)
                x_train = torch.tensor(x_train).float()

                time_test = torch.tensor(clinical.loc[test_ids, time_col].values.astype(float)).float()
                event_test = torch.tensor(clinical.loc[test_ids, event_col].values.astype(int)).bool()
                x_test = data.loc[test_ids].values.astype(float)
                x_test = scaler.fit_transform(x_test)
                x_test = torch.tensor(x_test).float()

                model = Model(x_train.shape[1])

                monitor = fit(model=model, x=x_train, event=event_train, time=time_train)
                y_hat = model(x_test)

                cindex = ConcordanceIndex()
                cindex_score = cindex(y_hat, event=event_test, time=time_test)

                score = {
                    'split_id': split['id'],
                    'target': target_name,
                    'model': model_name,
                    'cindex': cindex_score.item(),
                    'monitor': monitor
                }
                scores.append(score)

    pdat = [{k: v for k, v in i.items() if k in ['split_id', 'model', 'cindex']} for i in scores]
    pdat = pd.DataFrame(pdat)
    pdat['cindex'] = [float(v) for v in pdat.cindex]

    ax = sns.boxplot(data=pdat, x='model', y='cindex', fliersize=0)
    ax = sns.stripplot(data=pdat, x='model', y='cindex', color='black', ax=ax)
    ax.tick_params(axis='x', labelrotation=90)
    ax.figure.tight_layout()
    ax.figure.show()
    save_path = save_dir / f'{target_name}.png'
    ax.figure.savefig(save_path)
    print(f'Figure saved to: {save_path}')

# %%
# pdat = pd.DataFrame(scores[0]['monitor'])
# pdat['cindex'] = [float(v) for v in pdat.cindex]
# ax = sns.lineplot(data=pdat, x='epoch', y='cindex')
# ax.figure.show()