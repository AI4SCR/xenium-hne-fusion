import pandas as pd
import pytest

from xenium_hne_fusion.eval.stats import paired_t_tests


def test_paired_t_tests_pairs_models_by_metadata_path():
    runs = pd.DataFrame(
        [
            {
                'run_id': 'v0',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'test/pearson_mean': 0.20,
            },
            {
                'run_id': 'f0',
                'config.wandb.name': 'expr-token',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'test/pearson_mean': 0.30,
            },
            {
                'run_id': 'v1',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=1-seed=0.parquet',
                'test/pearson_mean': 0.25,
            },
            {
                'run_id': 'f1',
                'config.wandb.name': 'expr-token',
                'config.data.metadata_path': 'hescape/breast/outer=1-seed=0.parquet',
                'test/pearson_mean': 0.45,
            },
            {
                'run_id': 'v2',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=2-seed=0.parquet',
                'test/pearson_mean': 0.35,
            },
        ]
    )

    result = paired_t_tests(
        runs,
        metrics=['test/pearson_mean'],
        baseline='vision',
        candidates=['expr-token'],
    )

    row = result.iloc[0]
    assert row['metric'] == 'test/pearson_mean'
    assert row['baseline'] == 'vision'
    assert row['candidate'] == 'expr-token'
    assert row['n_pairs'] == 2
    assert row['mean_baseline'] == pytest.approx(0.225)
    assert row['mean_candidate'] == pytest.approx(0.375)
    assert row['mean_diff'] == pytest.approx(0.15)


def test_paired_t_tests_labels_model_variants():
    runs = pd.DataFrame(
        [
            {
                'run_id': 'v0',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'test/pearson_mean': 0.20,
            },
            {
                'run_id': 'v0-freeze',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'config.backbone.freeze_morph_encoder': True,
                'test/pearson_mean': 0.25,
            },
            {
                'run_id': 'v1',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=1-seed=0.parquet',
                'test/pearson_mean': 0.30,
            },
            {
                'run_id': 'v1-freeze',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=1-seed=0.parquet',
                'config.backbone.freeze_morph_encoder': True,
                'test/pearson_mean': 0.40,
            },
        ]
    )

    result = paired_t_tests(
        runs,
        metrics=['test/pearson_mean'],
        baseline='vision',
        candidates=['vision-freeze-morph'],
    )

    assert result.loc[0, 'candidate'] == 'vision-freeze-morph'


def test_paired_t_tests_rejects_duplicate_comparison_split_scores():
    runs = pd.DataFrame(
        [
            {
                'run_id': 'v0',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'config.backbone.fusion_stage': 'early',
                'test/pearson_mean': 0.20,
            },
            {
                'run_id': 'v0-duplicate',
                'config.wandb.name': 'vision',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'config.backbone.fusion_stage': 'late',
                'test/pearson_mean': 0.25,
            },
            {
                'run_id': 'f0',
                'config.wandb.name': 'expr-token',
                'config.data.metadata_path': 'hescape/breast/outer=0-seed=0.parquet',
                'test/pearson_mean': 0.30,
            },
        ]
    )

    with pytest.raises(AssertionError, match='Duplicate scores'):
        paired_t_tests(
            runs,
            metrics=['test/pearson_mean'],
            baseline='vision',
            candidates=['expr-token'],
        )
