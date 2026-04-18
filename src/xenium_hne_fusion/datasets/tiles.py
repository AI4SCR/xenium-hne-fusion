
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from ai4bmr_learn.datasets.items import Items


class TileDataset(Items):
    """
    Per-tile H&E image + tile-local target dataset.

    Each item must have a 'tile_dir' key pointing to a directory containing:
        tile.pt                     # uint8 CHW tensor
        expr-kernel_size=16.parquet # (n_tokens x n_genes) DataFrame
        cells.parquet               # optional per-cell table for cell type targets

    Args:
        target: prediction target, either tile-level expression or cell type counts.
        source_panel: expression genes to load into expr_tokens.
        target_panel: expression genes to aggregate into the target when target='expression'.
        include_image: load tile.pt.
        include_expr: load expr-kernel_size=16.parquet.
        target_transform: applied to the target tensor.
        image_transform: applied to uint8 CHW tensor.
        expr_transform: applied to float expr tensor after optional pooling.
        expr_pool: 'token' keeps (n_tokens, n_genes); 'tile' avg-pools to (n_genes,).
        cell_type_col: categorical cell type column to count when target='cell_types'.
    """

    def __init__(self, *,
                 target: Literal['cell_types', 'expression'],
                 source_panel: list[str] | None = None,
                 target_panel: list[str] | None = None,
                 include_image: bool = True,
                 include_expr: bool = True,
                 target_transform: Callable | None = None,
                 image_transform: Callable | None = None,
                 expr_transform: Callable | None = None,
                 expr_pool: Literal['token', 'tile'] = 'token',
                 cell_type_col: str = 'Level3_grouped',
                 **kwargs):

        super().__init__(**kwargs)

        self.target = target
        self.source_panel = source_panel
        self.target_panel = target_panel
        self.include_image = include_image
        self.include_expr = include_expr
        self.target_transform = target_transform
        self.image_transform = image_transform
        self.expr_transform = expr_transform
        self.expr_pool = expr_pool
        self.cell_type_col = cell_type_col

        assert target == 'expression' and target_panel is not None or target == 'cell_types', "target_panel must be specified when target is 'expression'"
        if target == 'expression' and source_panel is not None:
            assert target_panel is not None
            assert set(source_panel).isdisjoint(set(target_panel)), 'source_panel and target_panel must be disjoint'

    def __getitem__(self, idx) -> dict:
        item = deepcopy(self.items[idx])
        iid = item[self.id_key] if self.id_key else None
        tile_dir = Path(item['tile_dir'])

        if self.cache_dir is not None and self.has_cache(iid=iid):
            item = torch.load(self.get_cache_path(iid), weights_only=False)
            if not self.include_image:
                item['modalities'].pop('image', None)
            if not self.include_expr:
                item['modalities'].pop('expr_tokens', None)
        else:

            if self.include_expr or self.target == 'expression':
                expr = pd.read_parquet(tile_dir / f'expr-kernel_size=16.parquet')
                if 'token_index' in expr.columns:
                    expr = expr.drop(columns=['token_index'])

            # construct target
            if self.target == 'expression':
                assert self.target_panel is not None
                missing = sorted(set(self.target_panel) - set(expr.columns))
                assert not missing, f'missing target genes: {missing[:8]}'
                target = expr[self.target_panel].sum()
            elif self.target == 'cell_types':
                cell_types = pd.read_parquet(tile_dir / 'cells.parquet')
                assert cell_types[self.cell_type_col].dtype == 'category', f"{self.cell_type_col} must be categorical"
                target = cell_types[self.cell_type_col].value_counts().sort_index()
            else:
                raise ValueError(f"Unsupported target: {self.target}")

            target = torch.tensor(target.values, dtype=torch.float32)
            item['target'] = target

            modalities = {}
            if self.include_image:
                modalities['image'] = torch.load(tile_dir / 'tile.pt', weights_only=True)

            if self.include_expr:
                assert self.source_panel is not None, 'source_panel must be specified when include_expr=True'
                missing = sorted(set(self.source_panel) - set(expr.columns))
                assert not missing, f'missing source genes: {missing[:8]}'
                source = expr[self.source_panel]
                source = torch.tensor(source.values, dtype=torch.float32)
                modalities['expr_tokens'] = source

            item['modalities'] = modalities

        if self.include_expr and self.expr_pool == 'tile':
            item['modalities']['expr_tokens'] = item['modalities']['expr_tokens'].mean(dim=0)

        if self.target_transform is not None:
            item['target'] = self.target_transform(item['target'])

        if self.include_image and self.image_transform is not None:
            item['modalities']['image'] = self.image_transform(item['modalities']['image'])

        if self.include_expr and self.expr_transform is not None:
            item['modalities']['expr_tokens'] = self.expr_transform(item['modalities']['expr_tokens'])

        if self.metadata is not None:
            metadata_dict = self.metadata.loc[iid].to_dict()
            item['metadata'] = metadata_dict

        return item
