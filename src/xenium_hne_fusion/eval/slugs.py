from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from loguru import logger


SLUG_COLUMNS = ['stage', 'strategy', 'pool', 'learnable_gate', 'morph_encoder', 'expr_encoder', 'freeze_morph', 'freeze_expr']


@dataclass(frozen=True)
class SlugSpec:
    slug: str
    label: str
    order: int
    modality: str
    stage: str | None
    strategy: str | None
    pool: str | None
    learnable_gate: bool | None
    morph_encoder: str | None
    expr_encoder: str | None
    freeze_morph: bool | None = None
    freeze_expr: bool | None = None


def validate_slug_specs(specs: Mapping[str, SlugSpec]) -> None:
    orders = pd.Series([spec.order for spec in specs.values()])
    assert orders.is_unique, 'Duplicate slug orders'


def add_slugs(table: pd.DataFrame, specs: Mapping[str, SlugSpec]) -> pd.DataFrame:
    table = table.copy()
    table['slug'] = table.apply(lambda row: canonical_slug(row, specs), axis=1)
    missing_rows = table['slug'].isna()
    missing_names = table.loc[missing_rows, 'run_name'].astype(str)
    missing = sorted(missing_names[~missing_names.isin(specs)].unique())
    assert not missing, f'Runs missing from slug allowlist: {missing}'
    if missing_rows.any():
        dropped = sorted(missing_names.unique())
        logger.warning(f'Dropping W&B runs outside curated slug config: {dropped}')
        table = table.loc[~missing_rows].copy()
    return table


def canonical_slug(row: Mapping[str, Any], specs: Mapping[str, SlugSpec]) -> str | None:
    config_slug = _slug_from_config(row)
    if _has_slug_config(row):
        return config_slug if config_slug in specs else None

    names = [
        _value(row, 'run_name'),
        _value(row, 'config.wandb.name'),
    ]
    for name in names:
        if name in specs:
            return name

    return config_slug if config_slug in specs else None


def build_annotation_table(specs: Mapping[str, SlugSpec], slugs: list[str]) -> pd.DataFrame:
    assert slugs, 'No slugs to annotate'
    unknown = sorted(set(slugs) - set(specs))
    assert not unknown, f'Unknown slugs: {unknown}'
    duplicated = pd.Series(slugs)
    assert duplicated.is_unique, f'Duplicate plotted slugs: {duplicated[duplicated.duplicated()].tolist()}'

    rows = {
        slug: {column: getattr(specs[slug], column) for column in SLUG_COLUMNS}
        for slug in slugs
    }
    return pd.DataFrame(rows).loc[SLUG_COLUMNS]


def ordered_slugs(table: pd.DataFrame, specs: Mapping[str, SlugSpec]) -> list[str]:
    slugs = sorted(table['slug'].dropna().unique(), key=lambda slug: specs[slug].order)
    assert slugs, 'No allowlisted W&B runs to plot'
    return slugs


def ordered_labels(slugs: list[str], specs: Mapping[str, SlugSpec]) -> list[str]:
    return [specs[slug].label for slug in slugs]


def _slug_from_config(row: Mapping[str, Any]) -> str | None:
    fusion_strategy = _value(row, 'config.backbone.fusion_strategy')
    fusion_stage = _value(row, 'config.backbone.fusion_stage')
    morph_encoder = _value(row, 'config.backbone.morph_encoder_name')
    expr_encoder = _value(row, 'config.backbone.expr_encoder_name')
    expr_pool = _value(row, 'config.data.expr_pool')
    expr_token_pool = _value(row, 'config.backbone.expr_token_pool')
    learnable_gate = _value(row, 'config.backbone.learnable_gate')
    freeze_morph = _value(row, 'config.backbone.freeze_morph_encoder')
    freeze_expr = _value(row, 'config.backbone.freeze_expr_encoder')
    freeze_suffix = _freeze_suffix(freeze_morph, freeze_expr)

    if fusion_strategy is not None:
        if fusion_strategy != 'add':
            return None
        if fusion_stage == 'early':
            slug = 'early-fusion'
            if _is_true(learnable_gate):
                slug = f'{slug}-gate'
            return f'{slug}{freeze_suffix}'
        if fusion_stage == 'late':
            pool = 'tile' if expr_pool == 'tile' or expr_token_pool is None else 'token'
            slug = f'late-fusion-{pool}'
            if _is_true(learnable_gate):
                slug = f'{slug}-gate'
            return f'{slug}{freeze_suffix}'
        return None

    if morph_encoder is not None and expr_encoder is None:
        return f'vision{freeze_suffix}'
    if expr_encoder is not None and morph_encoder is None:
        base = 'expr-tile' if expr_pool == 'tile' else 'expr-token'
        return f'{base}{freeze_suffix}'
    return None


def _freeze_suffix(freeze_morph: Any, freeze_expr: Any) -> str:
    fm = _is_true(freeze_morph)
    fe = _is_true(freeze_expr)
    if fm and fe:
        return '-frozen'
    if fm:
        return '-morph-frozen'
    if fe:
        return '-expr-frozen'
    return ''


def _has_slug_config(row: Mapping[str, Any]) -> bool:
    keys = [
        'config.backbone.fusion_strategy',
        'config.backbone.fusion_stage',
        'config.backbone.morph_encoder_name',
        'config.backbone.expr_encoder_name',
        'config.data.expr_pool',
        'config.backbone.expr_token_pool',
        'config.backbone.freeze_morph_encoder',
        'config.backbone.freeze_expr_encoder',
    ]
    return any(_value(row, key) is not None for key in keys)


def _is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == 'true'
    return bool(value)


def _value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set)):
        return value
    if pd.isna(value):
        return None
    return value
