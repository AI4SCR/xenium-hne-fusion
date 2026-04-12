from pathlib import Path

import pandas as pd

from xenium_hne_fusion.eval import wandb as eval_wandb


class FakeSummary:
    def __init__(self, data):
        self._json_dict = data


class FakeRun:
    id = 'run-1'
    name = 'vision'
    state = 'finished'
    tags = ['lung']

    summary = FakeSummary({'test/pearson_mean': 0.4, 'test/spearman_mean': 0.3})
    config = {
        'data': {'name': 'beat'},
        'backbone': {'morph_encoder_name': 'vit_small_patch16_224', 'expr_encoder_name': None},
        'wandb': {'name': 'vision', 'tags': ['lung']},
    }


def test_load_project_runs_caches_pulled_wandb_table(monkeypatch, tmp_path: Path):
    calls = []

    def fake_fetch_runs(project, *, entity, filters):
        calls.append((project, entity, filters))
        return [FakeRun()]

    monkeypatch.setattr(eval_wandb, 'fetch_runs', fake_fetch_runs)

    first = eval_wandb.load_project_runs('project-a', entity='entity-a', cache_dir=tmp_path)
    second = eval_wandb.load_project_runs('project-a', entity='entity-a', cache_dir=tmp_path)

    assert calls == [('project-a', 'entity-a', {'state': 'finished'})]
    assert first.equals(second)
    assert first.loc[0, 'config.data.name'] == 'beat'
    assert first.loc[0, 'test/pearson_mean'] == 0.4
    assert (tmp_path / 'entity-a-project-a.parquet').exists()


def test_load_project_runs_refreshes_existing_cache(monkeypatch, tmp_path: Path):
    pd.DataFrame([{'run_id': 'cached'}]).to_parquet(tmp_path / 'entity-a-project-a.parquet')
    calls = []

    def fake_fetch_runs(project, *, entity, filters):
        calls.append(project)
        return [FakeRun()]

    monkeypatch.setattr(eval_wandb, 'fetch_runs', fake_fetch_runs)

    table = eval_wandb.load_project_runs('project-a', entity='entity-a', cache_dir=tmp_path, refresh=True)

    assert calls == ['project-a']
    assert table.loc[0, 'run_id'] == 'run-1'
