
import timm
import torch
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
_arcsinh= lambda x: torch.arcsinh(x)

def get_expr_encoder_and_transform(*, expr_encoder_name: str, input_dim: int | None = None, output_dim: int | None = None, source_panel: list[str] | None = None, **kws):
    match expr_encoder_name:
        case "mlp":
            expr_encoder_dim = output_dim
            expr_encoder = Head(input_dim=input_dim, output_dim=output_dim, **kws)
            expr_transform = _log1p
        case "geneformer":
            from xenium_hne_fusion.models.geneformer import Geneformer
            transform = lambda x: torch.expm1(x).int()
            expr_encoder = Geneformer(gene_names=source_panel, transform=transform, **kws)
            expr_transform = _log1p
            expr_encoder_dim = expr_encoder.embed_dim
        case None:
            expr_encoder = None
            expr_transform = _log1p
            expr_encoder_dim = None
        case _:
            raise ValueError(f"Unknown expr_encoder_name: {expr_encoder_name}")

    return expr_encoder, expr_transform, expr_encoder_dim

def get_morph_encoder_and_transform(*, morph_encoder_name: str, img_size: int = 224, **kws):
    if morph_encoder_name is not None and morph_encoder_name in timm.list_models():
        morph_encoder = timm.create_model(
            model_name=morph_encoder_name,
            pretrained=True,
            img_size=img_size,
            in_chans=3,
            num_classes=0,
            global_pool='',  # disable pooling and handle with global_pool in FusionModel
            **kws,
        )
        transform = get_timm_transform(morph_encoder)
        normalize = get_normalize_from_transform(transform)

        assert all(map(lambda x: x == 0.5, normalize.mean)), f"Expected mean 0.5, got {normalize.mean}"
        assert all(map(lambda x: x == 0.5, normalize.std)), f"Expected std 0.5, got {normalize.std}"

        # No spatial resize — tiles are assumed to be img_size × img_size already.
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )
        morph_encoder_dim = MODEL_EMBEDDING_DIMS.get(morph_encoder_name)

    elif morph_encoder_name in ['conch_v1.5', 'conch_v1.5_trunk']:
        import lazyslide as zs
        titan = zs.models.multimodal.Titan()

        model, transform = titan.conch, titan.conch_transform
        if morph_encoder_name == 'conch_v1.5_trunk':
            morph_encoder = model.trunk  # image encoder without CLIP proj head
            morph_encoder_dim = 1024
        else:
            morph_encoder = model  # full model with CLIP proj head
            morph_encoder_dim = 768

        transform = titan.get_transform()
        normalize = get_normalize_from_transform(transform)
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )

    elif morph_encoder_name == 'phikon':
        from xenium_hne_fusion.models.phikon import Phikon
        morph_encoder = Phikon()
        image_transform = morph_encoder.get_transform()
        morph_encoder_dim = morph_encoder.embed_dim

    elif morph_encoder_name == 'loki':
        from xenium_hne_fusion.models.loki import Loki
        # TODO: download utils for loki
        ckpt_path = ''
        morph_encoder = Loki(ckpt_path=ckpt_path)
        image_transform = morph_encoder.get_transform()
        morph_encoder_dim = morph_encoder.embed_dim

    elif morph_encoder_name == 'midnight':
        from xenium_hne_fusion.models.midnight import Midnight
        morph_encoder = Midnight()
        image_transform = morph_encoder.get_transform()
        morph_encoder_dim = morph_encoder.embed_dim

    else:
        morph_encoder = None
        image_transform = None
        morph_encoder_dim = None

    return morph_encoder, image_transform, morph_encoder_dim
