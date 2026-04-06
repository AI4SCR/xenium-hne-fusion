import torch
import pytest
from torch import nn

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.lit import RegressionLit
from xenium_hne_fusion.train.supervised import infer_head_input_dim, resolve_num_outputs, validate_task_config


def make_expression_cfg() -> Config:
    cfg = Config()
    cfg.task.target = 'expression'
    cfg.lit.target_key = 'target'
    cfg.data.source_panel = ['A', 'B']
    cfg.data.target_panel = ['C', 'D', 'E']
    return cfg


def make_cell_type_cfg() -> Config:
    cfg = Config()
    cfg.task.target = 'cell_types'
    cfg.head.output_dim = 39
    cfg.lit.target_key = 'target'
    return cfg


def test_expression_resolves_output_dim_from_target_panel():
    cfg = make_expression_cfg()

    validate_task_config(cfg)

    assert resolve_num_outputs(cfg) == 3


def test_expression_rejects_head_output_dim_override():
    cfg = make_expression_cfg()
    cfg.head.output_dim = 39

    with pytest.raises(AssertionError, match='cfg.head.output_dim'):
        validate_task_config(cfg)


def test_expression_requires_target_key_target():
    cfg = make_expression_cfg()
    cfg.lit.target_key = 'metadata.cell_type_target'

    with pytest.raises(AssertionError, match='cfg.lit.target_key'):
        validate_task_config(cfg)


def test_cell_type_prediction_requires_explicit_head_output_dim():
    cfg = make_cell_type_cfg()
    cfg.head.output_dim = None

    with pytest.raises(AssertionError, match='cfg.head.output_dim'):
        validate_task_config(cfg)


def test_cell_type_prediction_requires_target_key_target():
    cfg = make_cell_type_cfg()
    cfg.lit.target_key = 'metadata.cell_type_target'

    with pytest.raises(AssertionError, match='cfg.lit.target_key'):
        validate_task_config(cfg)


def test_cell_type_prediction_uses_explicit_head_output_dim():
    cfg = make_cell_type_cfg()

    validate_task_config(cfg)

    assert resolve_num_outputs(cfg) == 39


def test_infer_head_input_dim_for_late_concat_doubles_morph_dim():
    input_dim = infer_head_input_dim(
        fusion_stage='late',
        fusion_strategy='concat',
        morph_encoder_dim=384,
        expr_encoder_dim=384,
    )

    assert input_dim == 768


def test_regression_lit_reads_target_from_target_key():
    class DummyBackbone(nn.Module):
        def forward(self, batch):
            return batch['image']

    lit = RegressionLit(
        backbone=DummyBackbone(),
        head=nn.Linear(4, 39),
        num_outputs=39,
        batch_key='modalities',
        target_key='target',
        save_hparams=False,
    )

    batch = {
        'modalities': {'image': torch.randn(2, 4)},
        'target': torch.randn(2, 39),
    }

    _, y_hat, y, loss = lit.shared_step(batch, batch_idx=0)

    assert y.shape == (2, 39)
    assert y_hat.shape == (2, 39)
    assert loss.ndim == 0
