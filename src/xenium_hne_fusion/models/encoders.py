
from collections.abc import Callable
from dataclasses import dataclass

import timm
import torch
from torch import nn
from torchvision.transforms import v2

from xenium_hne_fusion.models.mlp import Head
from xenium_hne_fusion.transforms.utils import get_normalize_from_transform, get_timm_transform

MODEL_EMBEDDING_DIMS = {
    'vit_small_patch16_224': 384,
    'vit_base_patch16_224': 768,
    'conch_v1.5': 768,
    'conch_v1.5_trunk': 1024,
}


@dataclass
class EncoderSpec:
    encoder: nn.Module | None
    transform: Callable | None
    dim: int | None


def log1p_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x)


def expm1_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x).int()


def is_half(value: float) -> bool:
    return value == 0.5


def _image_transform_from_normalize(normalize: v2.Transform) -> v2.Transform:
    # No spatial resize — tiles are assumed to be img_size × img_size already.
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize])


def _assert_no_extra_kws(morph_encoder_name: str, kws: dict) -> None:
    assert not kws, f"Unexpected kws for morph_encoder_name={morph_encoder_name}: {kws}"


def _build_mlp_expr_encoder(*, expr_encoder_name: str, input_dim: int | None, output_dim: int | None, source_panel: list[str] | None, **kws) -> EncoderSpec:
    encoder = Head(input_dim=input_dim, output_dim=output_dim, **kws)
    return EncoderSpec(encoder, log1p_transform, output_dim)


def _build_timm_expr_encoder(*, expr_encoder_name: str, input_dim: int | None, output_dim: int | None, source_panel: list[str] | None, **kws) -> EncoderSpec:
    encoder = timm.create_model(
        model_name=expr_encoder_name,
        pretrained=True,
        img_size=224,
        in_chans=3,
        num_classes=0,
        global_pool='',  # disable pooling and handle with global_pool in FusionModel
        **kws,
    )
    encoder_dim = MODEL_EMBEDDING_DIMS.get(expr_encoder_name)
    encoder.patch_embed = nn.Linear(in_features=input_dim, out_features=encoder_dim)
    return EncoderSpec(encoder, log1p_transform, encoder_dim)


def _build_geneformer_encoder(*, expr_encoder_name: str, input_dim: int | None, output_dim: int | None, source_panel: list[str] | None, **kws) -> EncoderSpec:
    from xenium_hne_fusion.models.geneformer import Geneformer
    encoder = Geneformer(gene_names=source_panel, transform=expm1_transform, **kws)
    return EncoderSpec(encoder, log1p_transform, encoder.embed_dim)


def _build_residual_mlp_expr_encoder(*, expr_encoder_name: str, input_dim: int | None, output_dim: int | None, source_panel: list[str] | None, **kws) -> EncoderSpec:
    from xenium_hne_fusion.models.residual_mlp import ResidualMLP
    output_dim = output_dim or 384
    encoder = ResidualMLP(input_dim=input_dim, output_dim=output_dim, **kws)
    return EncoderSpec(encoder, log1p_transform, output_dim)


EXPR_ENCODER_REGISTRY: dict[str, Callable[..., EncoderSpec]] = {
    "mlp": _build_mlp_expr_encoder,
    "resmlp": _build_residual_mlp_expr_encoder,
    "vit_small_patch16_224": _build_timm_expr_encoder,
    "vit_base_patch16_224": _build_timm_expr_encoder,
    "geneformer": _build_geneformer_encoder,
}


def get_expr_encoder_and_transform(
    *,
    expr_encoder_name: str | None,
    input_dim: int | None = None,
    output_dim: int | None = None,
    source_panel: list[str] | None = None,
    **kws,
) -> EncoderSpec:
    if expr_encoder_name is None:
        return EncoderSpec(None, log1p_transform, None)
    assert expr_encoder_name in EXPR_ENCODER_REGISTRY, f"Unknown expr_encoder_name: {expr_encoder_name}"
    builder = EXPR_ENCODER_REGISTRY[expr_encoder_name]
    return builder(
        expr_encoder_name=expr_encoder_name,
        input_dim=input_dim,
        output_dim=output_dim,
        source_panel=source_panel,
        **kws,
    )


def _build_timm_morph_encoder(*, morph_encoder_name: str, img_size: int = 224, **kws) -> EncoderSpec:
    encoder = timm.create_model(
        model_name=morph_encoder_name,
        pretrained=True,
        img_size=img_size,
        in_chans=3,
        num_classes=0,
        global_pool='',  # disable pooling and handle with global_pool in FusionModel
        **kws,
    )
    transform = get_timm_transform(encoder)
    normalize = get_normalize_from_transform(transform)

    assert all(map(is_half, normalize.mean)), f"Expected mean 0.5, got {normalize.mean}"
    assert all(map(is_half, normalize.std)), f"Expected std 0.5, got {normalize.std}"

    image_transform = _image_transform_from_normalize(normalize)
    encoder_dim = MODEL_EMBEDDING_DIMS.get(morph_encoder_name)
    return EncoderSpec(encoder, image_transform, encoder_dim)


def _build_conch_morph_encoder(*, morph_encoder_name: str, **kws) -> EncoderSpec:
    _assert_no_extra_kws(morph_encoder_name, kws)
    import lazyslide as zs
    titan = zs.models.multimodal.Titan()

    model = titan.conch
    if morph_encoder_name == 'conch_v1.5_trunk':
        encoder = model.trunk  # image encoder without CLIP proj head
        encoder_dim = 1024
    else:
        encoder = model  # full model with CLIP proj head
        encoder_dim = 768

    normalize = get_normalize_from_transform(titan.get_transform())
    image_transform = _image_transform_from_normalize(normalize)
    return EncoderSpec(encoder, image_transform, encoder_dim)


def _build_phikon_morph_encoder(*, morph_encoder_name: str, **kws) -> EncoderSpec:
    _assert_no_extra_kws(morph_encoder_name, kws)
    from xenium_hne_fusion.models.phikon import Phikon
    encoder = Phikon()
    return EncoderSpec(encoder, encoder.get_transform(), encoder.embed_dim)


def _build_loki_morph_encoder(*, morph_encoder_name: str, **kws) -> EncoderSpec:
    from xenium_hne_fusion.models.loki import Loki
    # TODO: download utils for loki
    ckpt_path = kws.pop('ckpt_path', '')
    _assert_no_extra_kws(morph_encoder_name, kws)
    encoder = Loki(ckpt_path=ckpt_path)
    return EncoderSpec(encoder, encoder.get_transform(), encoder.embed_dim)


def _build_midnight_morph_encoder(*, morph_encoder_name: str, **kws) -> EncoderSpec:
    _assert_no_extra_kws(morph_encoder_name, kws)
    from xenium_hne_fusion.models.midnight import Midnight
    encoder = Midnight()
    return EncoderSpec(encoder, encoder.get_transform(), encoder.embed_dim)


MORPH_ENCODER_REGISTRY: dict[str, Callable[..., EncoderSpec]] = {
    "conch_v1.5": _build_conch_morph_encoder,
    "conch_v1.5_trunk": _build_conch_morph_encoder,
    "phikon": _build_phikon_morph_encoder,
    "loki": _build_loki_morph_encoder,
    "midnight": _build_midnight_morph_encoder,
}


def get_morph_encoder_and_transform(*, morph_encoder_name: str | None, img_size: int = 224, **kws) -> EncoderSpec:
    if morph_encoder_name is None:
        return EncoderSpec(None, None, None)

    if morph_encoder_name in timm.list_models():
        return _build_timm_morph_encoder(morph_encoder_name=morph_encoder_name, img_size=img_size, **kws)

    assert morph_encoder_name in MORPH_ENCODER_REGISTRY, f"Unknown morph_encoder_name: {morph_encoder_name}"
    builder = MORPH_ENCODER_REGISTRY[morph_encoder_name]
    return builder(morph_encoder_name=morph_encoder_name, **kws)
