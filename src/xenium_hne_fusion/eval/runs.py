from __future__ import annotations

import pandas as pd
from loguru import logger


_DEDUP_KEY_COLUMNS = [
    'config.wandb.name',
    'config.data.panel_path',
    'config.data.metadata_path',
    'config.backbone.fusion_strategy',
    'config.backbone.fusion_stage',
    'config.backbone.learnable_gate',
    'config.backbone.freeze_morph_encoder',
    'config.backbone.freeze_expr_encoder',
]


def keep_latest_per_group(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest run per experiment group (model + config dimensions).

    Paths are normalized to filename stems for cross-machine stability.
    Aligns with resubmit.py's make_group_key logic.
    """
    if len(runs) < 2:
        return runs

    order_col = _run_order_column(runs)
    ordered = runs.assign(_run_order=order_col).sort_values('_run_order', kind='stable', na_position='first')

    key_cols = [c for c in _DEDUP_KEY_COLUMNS if c in ordered.columns]
    normalized = _normalize_path_key_cols(ordered, key_cols)

    duplicated = normalized.duplicated(key_cols, keep=False)
    if not duplicated.any():
        return ordered.drop(columns='_run_order')

    for _, group in normalized.loc[duplicated].groupby(key_cols, dropna=False, sort=False):
        ids = group['run_id'].astype(str).tolist()
        kept = str(group.iloc[-1]['run_id'])
        logger.warning(f'Duplicate W&B runs for identical eval setting; keeping latest {kept}; duplicate run_ids={ids}')

    keep_idx = normalized.drop_duplicates(key_cols, keep='last').index
    result = ordered.loc[keep_idx].drop(columns='_run_order')
    logger.info(f'Deduplication: {len(runs)} → {len(result)} runs (dropped {len(runs) - len(result)} duplicates)')
    return result


def _normalize_path_key_cols(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in key_cols:
        if 'path' in col:
            df[col] = df[col].apply(_filename_stem)
    return df


def _filename_stem(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return None
        except (TypeError, ValueError):
            pass
    s = str(value).replace('\\', '/')
    filename = s.rsplit('/', 1)[-1]
    return filename.rsplit('.', 1)[0] if '.' in filename else filename


def _run_order_column(runs: pd.DataFrame) -> pd.Series:
    if 'run_created_at' in runs.columns:
        created_at = pd.to_datetime(runs['run_created_at'], errors='coerce', utc=True)
        if created_at.notna().any():
            return created_at
    if 'run_updated_at' in runs.columns:
        updated_at = pd.to_datetime(runs['run_updated_at'], errors='coerce', utc=True)
        if updated_at.notna().any():
            return updated_at
    return pd.Series(range(len(runs)), index=runs.index)
