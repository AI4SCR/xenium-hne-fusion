from __future__ import annotations

import timm
import torch
from loguru import logger
from torchvision.transforms import v2

from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.transforms.utils import get_normalize_from_transform, get_timm_transform

MODEL_EMBEDDING_DIMS = {
    'vit_small_patch16_224': 384,
    'vit_base_patch16_224': 768,
    'conch_v1.5': 768,
    'conch_v1.5_trunk': 1024,
}

_log1p = lambda x: torch.log1p(x)


def get_morph_encoder_and_transform(*, morph_encoder_name: str | None, **kws):
    if morph_encoder_name is None:
        return None, None, None

    if morph_encoder_name in timm.list_models():
        morph_encoder = timm.create_model(
            model_name=morph_encoder_name,
            pretrained=True,
            img_size=224,
            in_chans=3,
            num_classes=0,
            global_pool='',
            **kws,
        )
        transform = get_timm_transform(morph_encoder)
        normalize = get_normalize_from_transform(transform)
        assert all(v == 0.5 for v in normalize.mean), f'Expected mean 0.5, got {normalize.mean}'
        assert all(v == 0.5 for v in normalize.std), f'Expected std 0.5, got {normalize.std}'
        image_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize])
        morph_encoder_dim = MODEL_EMBEDDING_DIMS.get(morph_encoder_name)
        assert morph_encoder_dim is not None, f'Add {morph_encoder_name!r} to MODEL_EMBEDDING_DIMS'
        return morph_encoder, image_transform, morph_encoder_dim

    if morph_encoder_name in ('conch_v1.5', 'conch_v1.5_trunk'):
        import lazyslide as zs
        titan = zs.models.multimodal.Titan()
        if morph_encoder_name == 'conch_v1.5_trunk':
            morph_encoder, morph_encoder_dim = titan.conch.trunk, 1024
        else:
            morph_encoder, morph_encoder_dim = titan.conch, 768
        normalize = get_normalize_from_transform(titan.get_transform())
        image_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize])
        return morph_encoder, image_transform, morph_encoder_dim

    raise ValueError(f'Unknown morph_encoder_name: {morph_encoder_name!r}')


def get_expr_encoder_and_transform(*, expr_encoder_name: str | None,
                                   input_dim: int | None = None,
                                   output_dim: int | None = None, **kws):
    if expr_encoder_name is None:
        return None, _log1p, None

    if expr_encoder_name == 'mlp':
        expr_encoder = Head(input_dim=input_dim, output_dim=output_dim, **kws)
        return expr_encoder, _log1p, output_dim

    raise ValueError(f'Unknown expr_encoder_name: {expr_encoder_name!r}')
