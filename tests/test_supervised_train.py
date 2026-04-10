from pathlib import Path

import pandas as pd
import torch
import pytest
from torch import nn
from torchvision.transforms import v2

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.models.encoders import log1p_transform
from xenium_hne_fusion.models.fusion import FusionModel
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


def test_fusion_model_accepts_learnable_gate_for_add_fusion():
    model = FusionModel(
        morph_encoder=nn.Identity(),
        expr_encoder=nn.Identity(),
        morph_encoder_dim=4,
        expr_encoder_dim=4,
        fusion_strategy='add',
        fusion_stage='late',
        global_pool='avg',
        learnable_gate=True,
    )

    assert model.fusion_alpha is not None
    assert model.fusion_alpha.requires_grad


@pytest.mark.parametrize(
    ('fusion_strategy', 'morph_encoder', 'expr_encoder'),
    [
        ('concat', nn.Identity(), nn.Identity()),
        (None, nn.Identity(), None),
        (None, None, nn.Identity()),
    ],
)
def test_fusion_model_rejects_learnable_gate_outside_add_fusion(
    fusion_strategy: str | None,
    morph_encoder: nn.Module | None,
    expr_encoder: nn.Module | None,
):
    with pytest.raises(AssertionError, match='learnable_gate requires fusion_strategy=\"add\"'):
        FusionModel(
            morph_encoder=morph_encoder,
            expr_encoder=expr_encoder,
            morph_encoder_dim=4 if morph_encoder is not None else None,
            expr_encoder_dim=4 if expr_encoder is not None else None,
            fusion_strategy=fusion_strategy,
            fusion_stage='late' if fusion_strategy is not None else None,
            global_pool='avg',
            learnable_gate=True,
        )


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


def _write_supervised_tile_dir(
    tile_dir: Path,
    *,
    image: torch.Tensor | None = None,
    expr: pd.DataFrame | None = None,
    cell_types: list[str] | None = None,
) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    image = image if image is not None else torch.full((3, 2, 2), 255, dtype=torch.uint8)
    torch.save(image, tile_dir / "tile.pt")

    if expr is not None:
        expr.to_parquet(tile_dir / "expr-kernel_size=16.parquet", index=False)

    if cell_types is not None:
        pd.DataFrame(
            {
                "Level3_grouped": pd.Categorical(
                    cell_types,
                    categories=["tumor", "stroma"],
                    ordered=False,
                ),
            }
        ).to_parquet(tile_dir / "cells.parquet")


def _write_items_json(items_path: Path, tile_dir: Path) -> None:
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")


def test_supervised_style_expression_dataset_log1p_transforms_inputs_and_targets(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    _write_supervised_tile_dir(
        tile_dir,
        expr=pd.DataFrame(
            {
                "A": [0, 1, 3],
                "B": [2, 0, 1],
                "C": [4, 5, 6],
            }
        ),
    )
    items_path = tmp_path / "items.json"
    _write_items_json(items_path, tile_dir)

    ds = TileDataset(
        target="expression",
        source_panel=["A", "B"],
        target_panel=["C"],
        include_image=False,
        include_expr=True,
        target_transform=log1p_transform,
        expr_transform=log1p_transform,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]

    assert item["target"].dtype == torch.float32
    assert item["target"].shape == (1,)
    assert item["target"].tolist() == pytest.approx([torch.log1p(torch.tensor(15.0)).item()])
    assert item["modalities"]["expr_tokens"].dtype == torch.float32
    assert item["modalities"]["expr_tokens"].shape == (3, 2)
    assert torch.all(item["modalities"]["expr_tokens"] >= 0)
    assert torch.allclose(
        item["modalities"]["expr_tokens"],
        torch.log1p(torch.tensor([[0.0, 2.0], [1.0, 0.0], [3.0, 1.0]])),
    )


def test_supervised_style_cell_type_dataset_normalizes_image_and_log1p_transforms_target(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    _write_supervised_tile_dir(
        tile_dir,
        image=torch.tensor(
            [
                [[0, 255], [128, 64]],
                [[255, 0], [64, 128]],
                [[128, 64], [255, 0]],
            ],
            dtype=torch.uint8,
        ),
        cell_types=["tumor", "tumor", "stroma"],
    )
    items_path = tmp_path / "items.json"
    _write_items_json(items_path, tile_dir)

    image_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = TileDataset(
        target="cell_types",
        include_image=True,
        include_expr=False,
        target_transform=log1p_transform,
        image_transform=image_transform,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]

    assert item["target"].dtype == torch.float32
    assert item["target"].tolist() == pytest.approx(torch.log1p(torch.tensor([2.0, 1.0])).tolist())
    assert item["modalities"]["image"].dtype == torch.float32
    assert item["modalities"]["image"].shape == (3, 2, 2)
    assert float(item["modalities"]["image"].min()) >= -3.0
    assert float(item["modalities"]["image"].max()) <= 3.0
