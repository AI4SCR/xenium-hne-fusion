from pathlib import Path

import pandas as pd
import pytest
import torch

from xenium_hne_fusion.datasets.tiles import TileDataset


def _write_tile_dir(
    tile_dir: Path,
    *,
    feature_names: list[str] | None = None,
    cell_types: list[str] | None = None,
) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(12, dtype=torch.uint8).reshape(3, 2, 2), tile_dir / "tile.pt")

    if feature_names is not None:
        expr = pd.DataFrame(
            {
                "A": [1, 0, 2],
                "B": [0, 1, 3],
                "C": [4, 5, 6],
            },
            index=pd.Index([0, 1, 2], name="token_index"),
        )
        expr.to_parquet(tile_dir / "expr-kernel_size=16.parquet")

    if cell_types is not None:
        cells = pd.DataFrame(
            {
                "Level3_grouped": pd.Categorical(cell_types, categories=["tumor", "stroma"], ordered=False),
                "other_col": range(len(cell_types)),
            }
        )
        cells.to_parquet(tile_dir / "cells.parquet")


def test_tile_dataset_include_flags_control_modalities(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    _write_tile_dir(tile_dir, feature_names=["A"], cell_types=["tumor"])

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")

    ds = TileDataset(
        target="expression",
        source_panel=["A", "B"],
        target_panel=["C"],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]
    assert "image" not in item["modalities"]
    assert "expr_tokens" not in item["modalities"]
    assert item["target"].tolist() == [15.0]


def test_tile_dataset_transforms_are_applied(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    _write_tile_dir(tile_dir, feature_names=["A"], cell_types=["tumor"])

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")

    ds = TileDataset(
        target="expression",
        source_panel=["A", "B"],
        target_panel=["C"],
        include_image=True,
        include_expr=True,
        target_transform=lambda x: x + 10,
        image_transform=lambda x: x + 1,
        expr_transform=lambda x: x * 2,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]
    assert item["target"].tolist() == [25.0]
    assert torch.equal(item["modalities"]["image"], torch.arange(12, dtype=torch.uint8).reshape(3, 2, 2) + 1)
    assert item["modalities"]["expr_tokens"].tolist() == [[2.0, 0.0], [0.0, 2.0], [4.0, 6.0]]


def test_tile_dataset_respects_cell_type_col(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    tile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros((3, 2, 2), dtype=torch.uint8), tile_dir / "tile.pt")
    pd.DataFrame(
        {
            "custom_cell_type": pd.Categorical(["tumor", "stroma", "tumor"], categories=["tumor", "stroma"], ordered=False),
        }
    ).to_parquet(tile_dir / "cells.parquet")

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")

    ds = TileDataset(
        target="cell_types",
        cell_type_col="custom_cell_type",
        include_image=False,
        include_expr=False,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]
    assert item["target"].tolist() == [2.0, 1.0]
    assert "modalities" in item
    assert item["modalities"] == {}


def test_tile_dataset_loads_matching_metadata_for_selected_split(tmp_path: Path):
    fit_tile_dir = tmp_path / "S1" / "0"
    val_tile_dir = tmp_path / "S1" / "1"
    _write_tile_dir(fit_tile_dir, feature_names=["A"])
    _write_tile_dir(val_tile_dir, feature_names=["A"])

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [
            {"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(fit_tile_dir)},
            {"id": "S1_1", "sample_id": "S1", "tile_id": 1, "tile_dir": str(val_tile_dir)},
        ]
    ).to_json(items_path, orient="records")

    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "split": ["fit", "val"],
            "patient": ["P1", "P2"],
            "score": [1.5, 2.5],
        },
        index=pd.Index(["S1_0", "S1_1"], name="id"),
    ).to_parquet(metadata_path)

    ds = TileDataset(
        target="expression",
        source_panel=["A", "B"],
        target_panel=["C"],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        metadata_path=metadata_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    assert len(ds) == 1
    item = ds[0]
    assert item["id"] == "S1_0"
    assert item["metadata"] == {
        "split": "fit",
        "patient": "P1",
        "score": 1.5,
    }


def test_tile_dataset_ignores_token_index_expr_column(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    tile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(12, dtype=torch.uint8).reshape(3, 2, 2), tile_dir / "tile.pt")
    pd.DataFrame(
        {
            "token_index": [0, 1, 2],
            "A": [1, 0, 2],
            "B": [0, 1, 3],
            "C": [4, 5, 6],
        }
    ).to_parquet(tile_dir / "expr-kernel_size=16.parquet", index=False)

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")

    ds = TileDataset(
        target="expression",
        source_panel=["A", "B"],
        target_panel=["C"],
        include_image=False,
        include_expr=True,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    item = ds[0]
    assert item["target"].tolist() == [15.0]
    assert item["modalities"]["expr_tokens"].tolist() == [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]


def test_tile_dataset_raises_clear_error_for_missing_target_genes(tmp_path: Path):
    tile_dir = tmp_path / "S1" / "0"
    tile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros((3, 2, 2), dtype=torch.uint8), tile_dir / "tile.pt")
    pd.DataFrame(
        {
            "A": [1, 0, 2],
            "B": [0, 1, 3],
        }
    ).to_parquet(tile_dir / "expr-kernel_size=16.parquet", index=False)

    items_path = tmp_path / "items.json"
    pd.DataFrame(
        [{"id": "S1_0", "sample_id": "S1", "tile_id": 0, "tile_dir": str(tile_dir)}]
    ).to_json(items_path, orient="records")

    ds = TileDataset(
        target="expression",
        source_panel=["A"],
        target_panel=["C"],
        include_image=False,
        include_expr=False,
        items_path=items_path,
        split="fit",
        id_key="id",
    )
    ds.setup()

    with pytest.raises(AssertionError, match="missing target genes"):
        ds[0]
