import torch
from torch import nn
from glom import glom
from loguru import logger
from ai4bmr_learn.utils.pooling import pool
from typing import Literal


def _validate_config(
    morph_encoder,
    expr_encoder,
    fusion_strategy,
    fusion_stage,
    global_pool,
    morph_token_pool,
    expr_token_pool,
    learnable_gate,
):
    assert morph_encoder is not None or expr_encoder is not None, (
        'At least one of morph_encoder or expr_model should be provided.'
    )

    assert fusion_strategy in [
        None,
        'add',
        'concat',
    ], f'Fusion strategy has to be one of [None, add, concat], got {fusion_strategy}'

    assert global_pool in [
        None,
        'avg',
        'max',
        'token',
        'flatten',
    ], f'Global pool has to be one of avg/max/token, got {global_pool}'

    assert not (morph_encoder is None and global_pool == 'token'), f'If no morph_encoder is provided, global_pool cannot be `token`.'

    if morph_token_pool is not None:
        assert fusion_stage == 'late', f'morph_token_pool is only used for late fusion.'
    if expr_token_pool is not None:
        assert fusion_stage == 'late' or fusion_strategy is None, f'expr_token_pool is only used for late fusion or uni-modal expr encoder.'
    if learnable_gate:
        assert fusion_strategy == 'add', f'learnable_gate requires fusion_strategy="add" not {fusion_strategy}.'


_MISSING = object()  # NOTE: None can be a valid path in a dict, this we need sentinel value to detect missing keys


class FusionModel(nn.Module):
    def __init__(
            self,
            morph_encoder: nn.Module | None,
            expr_encoder: nn.Module | None,
            morph_encoder_dim: int | None = None,
            expr_encoder_dim: int | None = None,
            fusion_strategy: str | None = 'add',
            fusion_stage: str | None = 'early',
            use_proj: bool = False,
            use_modality_embed: bool = False,
            global_pool: str | None = 'token',
            expr_token_pool: str | None = None,
            morph_token_pool: str | None = None,
            morph_key: str = 'image',
            expr_key: str = 'expr_tokens',
            allow_unimodal_routes: bool = False,
            pos_embed_layer_name: str = '_pos_embed',
            freeze_morph_encoder: bool = False,
            freeze_expr_encoder: bool = False,
            learnable_gate: bool = False
    ):
        """
        Unified backbone for morphology-only, expression-only, and fusion models.

        Route semantics are inferred from the input batch:
        - `fusion`: both `morph_key` and `expr_key` are present
        - `morph_only`: only `morph_key` is present
        - `expr_only`: only `expr_key` is present

        Configuration semantics are independent from route inference:
        - `fusion_strategy is None` means the model is configured as genuinely unimodal,
          so exactly one encoder must be provided.
        - `fusion_strategy is not None` means the model is configured for fusion,
          so both encoders must be provided.

        `allow_unimodal_routes` only matters in the second case. It allows a
        fusion-configured model to accept batches that contain only one modality
        at runtime. It does not block intentionally unimodal models configured
        with `fusion_strategy=None`.
        """

        _validate_config(
            morph_encoder,
            expr_encoder,
            fusion_strategy,
            fusion_stage,
            global_pool,
            morph_token_pool,
            expr_token_pool,
            learnable_gate,
        )

        super().__init__()

        self.morph_encoder = morph_encoder
        self.expr_encoder = expr_encoder

        self.morph_key = morph_key
        self.expr_key = expr_key
        self.allow_unimodal_routes = allow_unimodal_routes

        self.fusion_strategy = fusion_strategy
        self.fusion_stage = fusion_stage

        if fusion_strategy is None:
            assert (self.expr_encoder is None) != (self.morph_encoder is None), (
                'If fusion_strategy is None, then only one of expr_encoder or morph_encoder should be provided.'
            )
        else:
            assert self.expr_encoder is not None and self.morph_encoder is not None, (
                'If fusion_strategy is not None, then both expr_encoder and morph_encoder should be provided.'
            )

        self.epsilon = 1e-5  # needed for token normalization

        # Learnable residual scale for "add" fusion. tanh(0) = 0, so the
        # learnable-gate path starts from the pure morph baseline.
        self.fusion_alpha = nn.Parameter(torch.zeros(1), requires_grad=learnable_gate) if fusion_strategy == "add" else None

        self.pos_embed_layer_name = pos_embed_layer_name

        self.expr_to_morph_proj: nn.Module | None = None
        if (self.expr_encoder is not None) and (self.morph_encoder is not None):
            if use_proj:
                # TODO: do we need to activate?
                self.expr_to_morph_proj = nn.Linear(expr_encoder_dim, morph_encoder_dim)
                # self.expr_norm = nn.LayerNorm(self.morph_dim)  # TODO: check if this is needed
            elif fusion_strategy == 'add':
                assert expr_encoder_dim == morph_encoder_dim, (
                    'If `use_proj` is False, then expr_encoder_dim must be equal to morph_encoder_dim.'
                )
            else:
                logger.warning('No projection layer used for expr to morph dimension alignment.')
                assert use_modality_embed is False and expr_encoder_dim == morph_encoder_dim, f'Modality embedding requires same embedding dimension. Use `use_proj=True`.'

        if freeze_morph_encoder and self.morph_encoder is not None:
            self.morph_encoder.requires_grad_(False)

        if freeze_expr_encoder and self.expr_encoder is not None:
            self.expr_encoder.requires_grad_(False)

        self.global_pool = global_pool
        self.expr_token_pool = expr_token_pool
        self.morph_token_pool = morph_token_pool

        embed_dim = morph_encoder_dim or expr_encoder_dim
        self.use_modality_embed = use_modality_embed
        self.modality_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.ln_morph = nn.LayerNorm(embed_dim)
        self.ln_expr = nn.LayerNorm(embed_dim)

    def forward_morph(self, images: torch.Tensor) -> torch.Tensor:
        return self.morph_encoder(images)

    def get_fusion_gate(self) -> torch.Tensor:
        assert self.fusion_alpha is not None, 'fusion_alpha'
        if self.fusion_alpha.requires_grad:
            return torch.tanh(self.fusion_alpha)
        return torch.ones_like(self.fusion_alpha)

    def forward_late_fusion(self, *, morph_features: torch.Tensor, expr_features: torch.Tensor) -> torch.Tensor:

        assert morph_features is not None and expr_features is not None
        assert morph_features.shape == expr_features.shape, f'morph_features and expr_features must have the same shape for late fusion, got {morph_features.shape} and {expr_features.shape}'

        expr_features = self.ln_expr(expr_features)
        morph_features = self.ln_morph(morph_features)

        match self.fusion_strategy:
            case 'add':
                x = morph_features + self.get_fusion_gate() * expr_features
            case 'concat':
                x = torch.cat([morph_features, expr_features], dim=1)
            case _:
                raise ValueError(f'Unknown fusion strategy {self.fusion_strategy}')
        return x

    def forward_early_fusion(self, *, morph_tokens: torch.Tensor | None, expr_tokens: torch.Tensor | None = None) -> torch.Tensor:
        # NOTE: this is follows the features_forward pass from `timmm.models.vision_transformer.py`

        # apply mask
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L213

        assert morph_tokens is not None and expr_tokens is not None
        assert morph_tokens.shape == expr_tokens.shape, f'morph_features and expr_features must have the same shape for late fusion, got morph: {morph_tokens.shape} and expr: {expr_tokens.shape}'

        expr_scaled = self.normalize_expr_to_morph(morph_tokens=morph_tokens, expr_tokens=expr_tokens)

        # FUSION
        # TODO:
        #  - add gating similar to Flamingo models
        #  - how to embed the modality?
        match self.fusion_strategy:
            case 'add':
                x = morph_tokens + self.get_fusion_gate() * expr_scaled
                x = getattr(self.morph_encoder, self.pos_embed_layer_name)(x)

            case 'concat':
                morph_tokens = getattr(self.morph_encoder, self.pos_embed_layer_name)(morph_tokens)

                # we use the same positional embedding for expr_tokens
                # _pos_embed adds a cls token to the input so here we need to remove it
                expr_tokens = getattr(self.morph_encoder, self.pos_embed_layer_name)(expr_tokens)
                expr_tokens = expr_tokens[:, 1:]
                if self.use_modality_embed:
                    expr_tokens = expr_tokens + self.modality_token
                x = torch.cat([morph_tokens, expr_tokens], dim=1)

            case _:
                raise ValueError(f'Unknown fusion strategy {self.fusion_strategy}')

        # TODO: check how we want handle the patch_drop -> do not support ATM
        assert isinstance(self.morph_encoder.patch_drop, nn.Identity)

        x = self.morph_encoder.patch_drop(x)
        x = self.morph_encoder.norm_pre(x)

        for block in self.morph_encoder.blocks:
            x = block(x)
        return self.morph_encoder.norm(x)


    def normalize_expr_to_morph(self, morph_tokens: torch.Tensor, expr_tokens: torch.Tensor):
        """
        Given [B, N, D] morph and expr tokens:
        Scales each expr token to match the norm of the corresponding morph token
        """
        # TODO: USE LEARNED NORMALIZATION INSTEAD OF PER-TOKEN NORM MATCHING, USE LAYERNORM, RMSNORM?
        # compute per-token norms
        dtype = expr_tokens.dtype
        morph_norms = morph_tokens.float().norm(dim=-1, keepdim=True)  # [B, N, 1]
        expr_norms = expr_tokens.float().norm(dim=-1, keepdim=True).clamp(min=self.epsilon)

        # scale expr to match morph norm
        nonzero_mask = expr_norms.squeeze(-1) > self.epsilon
        expr_scaled = torch.zeros_like(expr_tokens, dtype=dtype)
        expr_scaled[nonzero_mask] = (
                expr_tokens[nonzero_mask] * (morph_norms[nonzero_mask] / expr_norms[nonzero_mask])
        ).to(dtype)

        return expr_scaled.to(dtype)

    def forward_expr_tokens(self, expr_tokens: torch.Tensor):
        expr_tokens = self.expr_encoder(expr_tokens)

        if self.expr_to_morph_proj is not None:
            expr_tokens = self.expr_to_morph_proj(expr_tokens)  # (B, num_transcripts_tokens, morph_dim)
            # expr_tokens = self.expr_norm(expr_tokens)  # TODO: should we norm?

        return expr_tokens

    def patchify(self, image: torch.Tensor):
        _, _, H, W = image.shape
        assert H == W

        morph_tokens = self.morph_encoder.patch_embed(image)  # (B, num_image_tokens, embed_dim)
        return morph_tokens

    def infer_route(self, batch: dict) -> Literal['fusion', 'morph_only', 'expr_only']:
        has_morph = glom(batch, self.morph_key, default=_MISSING) is not _MISSING
        has_expr = glom(batch, self.expr_key, default=_MISSING) is not _MISSING

        if has_morph and has_expr:
            return 'fusion'
        if has_morph:
            return 'morph_only'
        if has_expr:
            return 'expr_only'
        raise ValueError(f'Could not infer route: neither {self.morph_key} nor {self.expr_key} is present.')

    def validate_route(self, route: Literal['fusion', 'morph_only', 'expr_only']) -> None:
        if route == 'fusion':
            if self.fusion_strategy is None:
                raise ValueError('Inferred route `fusion` for a uni-modal configured model.')
            if self.morph_encoder is None or self.expr_encoder is None:
                raise ValueError('Fusion route requires both morph and expr encoders.')
            return

        if route == 'morph_only':
            if self.morph_encoder is None:
                raise ValueError('Inferred route `morph_only`, but no morph encoder is configured.')
        elif route == 'expr_only':
            if self.expr_encoder is None:
                raise ValueError('Inferred route `expr_only`, but no expr encoder is configured.')
        else:
            raise ValueError(f'Unknown route `{route}`.')

        if self.fusion_strategy is not None:
            if not self.allow_unimodal_routes:
                raise ValueError(
                    f"Inferred route '{route}' for fusion-configured model, but allow_unimodal_routes=False."
                )
            if self.fusion_stage == 'late' and self.fusion_strategy == 'concat':
                raise ValueError(
                    f"Unimodal route '{route}' is not supported for fusion_stage='late', fusion_strategy='concat'."
                )
            return

        if route == 'morph_only' and self.expr_encoder is not None and self.morph_encoder is None:
            raise ValueError('Inferred route `morph_only` for an expr-only configured model.')
        if route == 'expr_only' and self.morph_encoder is not None and self.expr_encoder is None:
            raise ValueError('Inferred route `expr_only` for a morph-only configured model.')

    def forward(self, batch: dict):
        route = self.infer_route(batch)
        self.validate_route(route)

        # FUSION
        if route == 'fusion':
            images = glom(batch, self.morph_key)
            expr_tokens = glom(batch, self.expr_key)

            expr_tokens = self.forward_expr_tokens(expr_tokens=expr_tokens)

            if self.fusion_stage == 'early':
                morph_tokens = self.patchify(images)  # (B, num_tokens, morph_dim)
                features = self.forward_early_fusion(morph_tokens=morph_tokens, expr_tokens=expr_tokens)

            elif self.fusion_stage == 'late':
                morph_features = self.forward_morph(images)

                morph_features = pool(morph_features, strategy=self.morph_token_pool)
                expr_features = pool(expr_tokens, strategy=self.expr_token_pool)

                features = self.forward_late_fusion(morph_features=morph_features, expr_features=expr_features)

        # UNI-MODAL
        if route == 'morph_only':
            images = glom(batch, self.morph_key)
            features = self.forward_morph(images)
        elif route == 'expr_only':
            expr_tokens = glom(batch, self.expr_key)
            expr_tokens = pool(expr_tokens, strategy=self.expr_token_pool)
            features = self.forward_expr_tokens(expr_tokens=expr_tokens)

        return pool(features, strategy=self.global_pool)
