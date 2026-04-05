from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from ai4bmr_learn.datasets.items import Items


class TileDataset(Items):
    """
    Per-tile H&E image + expression token dataset.

    Each item must have a 'tile_dir' key pointing to a directory containing:
        tile.pt                          # uint8 CHW tensor
        expr-kernel_size={k}.parquet     # (n_tokens × n_genes) DataFrame

    Args:
        kernel_size: sub-tile size used when computing expression tokens.
        panel: gene subset to select from expr columns (all if None).
        include_image: load tile.pt.
        include_expr: load expr parquet.
        image_transform: applied to uint8 CHW tensor.
        expr_transform: applied to float expr tensor after optional pooling.
        expr_pool: 'token' keeps (n_tokens, n_genes); 'tile' avg-pools to (n_genes,).
    """

    def __init__(self, *,
                 kernel_size: int = 16,
                 panel: list[str] | None = None,
                 target_panel: list[str] | None = None,
                 include_image: bool = True,
                 include_expr: bool = True,
                 image_transform: Callable | None = None,
                 expr_transform: Callable | None = None,
                 expr_pool: Literal['token', 'tile'] = 'token',
                 cell_types_kernel_size: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.panel = panel
        self.target_panel = target_panel
        self.include_image = include_image
        self.include_expr = include_expr
        self.image_transform = image_transform
        self.expr_transform = expr_transform
        self.expr_pool = expr_pool
        self.cell_types_kernel_size = cell_types_kernel_size

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

            if self.include_expr or self.target_panel is not None:
                expr_raw = pd.read_parquet(tile_dir / f'expr-kernel_size={self.kernel_size}.parquet')

            if self.include_expr:
                expr = expr_raw[self.panel] if self.panel is not None else expr_raw
                expr_t = torch.tensor(expr.values, dtype=torch.float32)
                if self.expr_pool == 'tile':
                    expr_t = expr_t.mean(dim=0)
                modalities['expr_tokens'] = expr_t

            if self.target_panel is not None:
                target_t = torch.tensor(expr_raw[self.target_panel].values, dtype=torch.float32)
                item['target'] = target_t.mean(dim=0)  # avg-pool to (n_target_genes,)

            if self.cell_types_kernel_size is not None:
                ct = pd.read_parquet(tile_dir / f'cell_types-kernel_size={self.cell_types_kernel_size}.parquet')
                item['target'] = torch.tensor(ct.values, dtype=torch.float32).mean(dim=0)

            item['modalities'] = modalities

        if self.image_transform is not None and 'image' in item.get('modalities', {}):
            item['modalities']['image'] = self.image_transform(item['modalities']['image'])

        if self.expr_transform is not None and 'expr_tokens' in item.get('modalities', {}):
            item['modalities']['expr_tokens'] = self.expr_transform(item['modalities']['expr_tokens'])

        if self.metadata is not None:
            import numpy as np
            item['metadata'] = {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in self.metadata.loc[iid].items()
            }

        return item
