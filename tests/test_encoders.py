import pytest

from xenium_hne_fusion.models.encoders import (
    EncoderSpec,
    get_expr_encoder_and_transform,
    get_morph_encoder_and_transform,
)


def test_expr_encoder_none_returns_empty_spec():
    spec = get_expr_encoder_and_transform(expr_encoder_name=None)
    assert isinstance(spec, EncoderSpec)
    assert spec.encoder is None
    assert spec.dim is None
    assert spec.transform is not None


def test_expr_encoder_mlp_builds_head_with_output_dim():
    spec = get_expr_encoder_and_transform(expr_encoder_name="mlp", input_dim=10, output_dim=32)
    assert spec.encoder is not None
    assert spec.dim == 32


def test_expr_encoder_unknown_name_raises():
    with pytest.raises(AssertionError, match="Unknown expr_encoder_name"):
        get_expr_encoder_and_transform(expr_encoder_name="bogus")


def test_morph_encoder_none_returns_empty_spec():
    spec = get_morph_encoder_and_transform(morph_encoder_name=None)
    assert isinstance(spec, EncoderSpec)
    assert spec.encoder is None
    assert spec.transform is None
    assert spec.dim is None


def test_morph_encoder_unknown_name_raises():
    with pytest.raises(AssertionError, match="Unknown morph_encoder_name"):
        get_morph_encoder_and_transform(morph_encoder_name="bogus")
