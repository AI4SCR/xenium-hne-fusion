# %%
from dotenv import load_dotenv

assert load_dotenv(override=True)

from pathlib import Path
from loguru import logger
import pandas as pd
from xenium_hne_fusion.utils.getters import get_managed_paths
import pyarrow.parquet as pq
from tqdm import tqdm

import json
import numpy as np

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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

data = np.log1p(data)

splits = json.load(Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/metadata_b_splits.json').open())
registry = {
    'logistic_regression': LogisticRegression(max_iter=10_000),
    'random_forest': RandomForestClassifier(),
    'svc': SVC(),
}
target = '7b'
scores = []
for model_name in registry.keys():
    for split in splits:

        model = registry[model_name]

        train_ids = split['train_ids']
        test_ids = split['test_ids']

        x_train = data.loc[train_ids]
        y_train = clinical.loc[train_ids, target]

        x_test = data.loc[test_ids]
        y_test = clinical.loc[test_ids, target]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit(X=x_train, y=y_train)
        y_hat = model.predict(X=x_test)

        score = {
            'split_id': split['id'],
            'balanced_accuracy': balanced_accuracy_score(y_true=y_test, y_pred=y_hat),
            'model': model_name
        }
        scores.append(score)

# %%
import seaborn as sns

scores = pd.DataFrame(scores)
ax = sns.boxplot(x='model', y='balanced_accuracy', data=scores, fliersize=0)
ax = sns.stripplot(x='model', y='balanced_accuracy', color='black', data=scores, ax=ax)
ax.set_title(target)
ax.figure.show()
save_path = save_dir / f'{target}.png'
ax.figure.savefig(save_path)
print(f'Figure saved to: {save_path}')
