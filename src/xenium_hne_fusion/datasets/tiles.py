from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from ai4bmr_learn.datasets.items import Items


class TileDataset(Items):
    """
    Dataset for per-tile H&E image + expression token data.

    Each item must have a 'tile_dir' key pointing to:
        tile.pt                          # uint8 CHW tensor
        expr-kernel_size={k}.parquet     # (n_tokens × n_genes) DataFrame

    Args:
        kernel_size: sub-tile kernel size used when building the expr parquet.
        panel: gene columns to select (all if None).
        include_image: load tile.pt.
        include_expr: load expr parquet.
        image_transform: applied to uint8 CHW image tensor.
        expr_transform: applied to float expr tensor (after optional pooling).
        expr_pool: 'token' keeps (n_tokens, n_genes); 'tile' avg-pools to (n_genes,).
    """

    def __init__(self, *,
                 kernel_size: int = 16,
                 panel: list[str] | None = None,
                 include_image: bool = True,
                 include_expr: bool = True,
                 image_transform: Callable | None = None,
                 expr_transform: Callable | None = None,
                 expr_pool: Literal['token', 'tile'] = 'token',
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.panel = panel
        self.include_image = include_image
        self.include_expr = include_expr
        self.image_transform = image_transform
        self.expr_transform = expr_transform
        self.expr_pool = expr_pool

    def __getitem__(self, idx) -> dict:
        item = deepcopy(self.items[idx])
        iid = item[self.id_key] if self.id_key else None
        tile_dir = Path(item['tile_dir'])

        if self.cache_dir is not None and self.has_cache(iid=iid):
            item = torch.load(self.get_cache_path(iid), weights_only=False)
        else:
            modalities = {}

            if self.include_image:
                modalities['image'] = torch.load(tile_dir / 'tile.pt', weights_only=True)

            if self.include_expr:
                expr = pd.read_parquet(tile_dir / f'expr-kernel_size={self.kernel_size}.parquet')
                if self.panel is not None:
                    expr = expr[self.panel]
                expr_t = torch.from_numpy(expr.values).float()
                if self.expr_pool == 'tile':
                    expr_t = expr_t.mean(dim=0)
                modalities['expr_tokens'] = expr_t

            item['modalities'] = modalities

        if self.image_transform is not None and 'image' in item.get('modalities', {}):
            item['modalities']['image'] = self.image_transform(item['modalities']['image'])

        if self.expr_transform is not None and 'expr_tokens' in item.get('modalities', {}):
            item['modalities']['expr_tokens'] = self.expr_transform(item['modalities']['expr_tokens'])

        if self.metadata is not None:
            item['metadata'] = self.metadata.loc[iid].to_dict()

        return item
