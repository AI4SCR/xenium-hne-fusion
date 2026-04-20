import pandas as pd

from xenium_hne_fusion.eval import plotting
from xenium_hne_fusion.eval.plotting import _configuration_ids, _ordered_configs


def test_ordered_configs_sort_by_mean_metric_across_splits():
    data = pd.DataFrame(
        [
            {'model': 'fusion', 'stage': 'late', 'split': 'outer=0', 'score': 0.9},
            {'model': 'fusion', 'stage': 'late', 'split': 'outer=1', 'score': 0.7},
            {'model': 'fusion', 'stage': 'early', 'split': 'outer=0', 'score': 0.4},
            {'model': 'fusion', 'stage': 'early', 'split': 'outer=1', 'score': 0.2},
        ]
    )
    data['config_id'] = _configuration_ids(data, ['model', 'stage'])

    ordered = _ordered_configs(data, metric='score', sort_by_score=True)

    assert ordered == ['fusion | late', 'fusion | early']


def test_ordered_configs_can_sort_by_name():
    data = pd.DataFrame(
        [
            {'model': 'fusion', 'stage': 'late', 'score': 0.9},
            {'model': 'fusion', 'stage': 'early', 'score': 0.2},
        ]
    )
    data['config_id'] = _configuration_ids(data, ['model', 'stage'])

    ordered = _ordered_configs(data, metric='score', sort_by_score=False)

    assert ordered == ['fusion | early', 'fusion | late']


def test_build_parameter_table_slugs_expression_encoder_names():
    data = pd.DataFrame(
        [
            {'config_id': 'mlp-config', 'config.backbone.expr_encoder_name': 'mlp'},
            {'config_id': 'gf-config', 'config.backbone.expr_encoder_name': 'geneformer'},
        ]
    )

    table = plotting._build_parameter_table(
        data,
        ['mlp-config', 'gf-config'],
        parameter_columns=['config.backbone.expr_encoder_name'],
    )

    assert table.loc['expr_encoder'].to_dict() == {'mlp-config': 'mlp', 'gf-config': 'gf'}
    assert set(table.loc['expr_encoder']) <= set(plotting.ANNOTATION_PALETTES['expr_encoder'])


def test_boolean_annotation_palettes_do_not_use_missing_color():
    for row in ['learnable_gate', 'freeze_morph', 'freeze_expr']:
        palette = plotting.ANNOTATION_PALETTES[row]
        assert palette['False'] == '#F5D2D2'
        assert palette['True'] == '#B0C4DE'
        assert plotting.NA_COLOR not in palette.values()
