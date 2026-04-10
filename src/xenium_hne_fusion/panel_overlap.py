import json
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from xenium_hne_fusion.config import ArtifactsConfig
from xenium_hne_fusion.metadata import normalize_sample_metadata
from xenium_hne_fusion.processing import load_feature_universe
from xenium_hne_fusion.utils.getters import apply_filter, get_managed_paths

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_items_with_metadata_from_frame(items: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    assert not items.empty, 'No items provided'
    metadata = normalize_sample_metadata(pd.read_parquet(metadata_path))

    columns = ['sample_id']
    if 'organ' in metadata.columns:
        columns.append('organ')
    metadata = metadata[columns].drop_duplicates()
    return items.merge(metadata, on='sample_id', how='left')


def load_items_with_metadata(items_path: Path, metadata_path: Path) -> pd.DataFrame:
    items = pd.DataFrame(json.loads(items_path.read_text()))
    assert not items.empty, f'No items found: {items_path}'
    return load_items_with_metadata_from_frame(items, metadata_path)


def _resolve_feature_universe_path(tile_dir: Path) -> Path:
    feature_universe_path = tile_dir.parent.parent / 'feature_universe.txt'
    assert feature_universe_path.exists(), f'Missing feature universe: {feature_universe_path}'
    return feature_universe_path


def collect_sample_summaries(items: pd.DataFrame) -> list[dict]:
    summaries: list[dict] = []
    group_cols = ['sample_id', 'tile_id'] if 'tile_id' in items.columns else ['sample_id']
    sorted_items = items.sort_values(group_cols)

    for sample_id, group in sorted_items.groupby('sample_id', sort=True):
        feature_universe_paths = {
            _resolve_feature_universe_path(Path(tile_dir))
            for tile_dir in group['tile_dir']
        }
        assert len(feature_universe_paths) == 1, f'Mixed feature universes for sample: {sample_id}'
        feature_universe_path = next(iter(feature_universe_paths))
        genes = tuple(load_feature_universe(feature_universe_path))
        organ = None if 'organ' not in group.columns else group['organ'].iloc[0]
        summaries.append(
            {
                'sample_id': sample_id,
                'organ': organ,
                'num_tiles': len(group),
                'genes': genes,
            }
        )

    summaries.sort(key=lambda row: ((row['organ'] is None), str(row['organ']), row['sample_id']))
    assert summaries, 'No sample summaries collected'
    return summaries


def select_artifact_items(
    artifacts_cfg: ArtifactsConfig,
    *,
    items_path: Path,
    metadata_path: Path,
    stats_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    items = pd.DataFrame(json.loads(items_path.read_text()))
    assert not items.empty, f'No items found: {items_path}'

    if artifacts_cfg.items.filter.organs is not None:
        metadata = normalize_sample_metadata(pd.read_parquet(metadata_path))
        assert 'organ' in metadata.columns, 'metadata.organ is required'
        allowed = set(metadata.loc[metadata['organ'].isin(artifacts_cfg.items.filter.organs), 'sample_id'])
        items = items[items['sample_id'].isin(allowed)]

    sample_ids = sorted(items['sample_id'].unique().tolist())
    sample_id_set = set(sample_ids)
    missing_include_ids: list[str] = []
    if artifacts_cfg.items.filter.include_ids is not None:
        include_ids = artifacts_cfg.items.filter.include_ids
        missing_include_ids = sorted(set(include_ids) - sample_id_set)
        keep_ids = [sample_id for sample_id in include_ids if sample_id in sample_id_set]
        items = items[items['sample_id'].isin(keep_ids)]
    elif artifacts_cfg.items.filter.exclude_ids is not None:
        exclude_ids = set(artifacts_cfg.items.filter.exclude_ids)
        items = items[~items['sample_id'].isin(exclude_ids)]

    stats = pd.read_parquet(stats_path)
    keep_item_ids = set(stats.index[apply_filter(stats, artifacts_cfg.items)])
    items = items[items['id'].isin(keep_item_ids)].copy()
    assert not items.empty, f'No items left after filtering: {artifacts_cfg.items.name}'
    return items, missing_include_ids


def compute_pairwise_overlap(sample_summaries: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for left in sample_summaries:
        left_set = set(left['genes'])
        for right in sample_summaries:
            right_set = set(right['genes'])
            intersection = left_set & right_set
            union = left_set | right_set
            rows.append(
                {
                    'left': left['sample_id'],
                    'right': right['sample_id'],
                    'intersection': len(intersection),
                    'union': len(union),
                    'jaccard': len(intersection) / len(union) if union else 1.0,
                }
            )
    overlap = pd.DataFrame(rows)
    assert len(overlap) == len(sample_summaries) ** 2, 'Incomplete overlap matrix'
    return overlap


def build_overlap_report(sample_summaries: list[dict], overlap: pd.DataFrame, missing_include_ids: list[str]) -> str:
    lines = ['Sample feature universes']
    if missing_include_ids:
        lines.append(f"- missing_configured_samples={missing_include_ids}")
    for summary in sample_summaries:
        lines.append(
            f"- {summary['sample_id']}: organ={summary['organ']} genes={len(summary['genes'])} "
            f"tiles={summary['num_tiles']}"
        )

    lines.append('')
    lines.append('Pairwise overlap')
    sample_ids = [summary['sample_id'] for summary in sample_summaries]
    for i, left in enumerate(sample_ids):
        for right in sample_ids[i + 1:]:
            row = overlap[(overlap['left'] == left) & (overlap['right'] == right)].iloc[0]
            lines.append(
                f"- {row['left']} vs {row['right']}: intersection={row['intersection']} "
                f"union={row['union']} jaccard={row['jaccard']:.3f}"
            )
    return '\n'.join(lines)


def plot_pairwise_overlap(sample_summaries: list[dict], overlap: pd.DataFrame, output_path: Path) -> Path:
    ordered_ids = [summary['sample_id'] for summary in sample_summaries]
    labels = [
        summary['sample_id'] if summary['organ'] is None else f"{summary['sample_id']}\n{summary['organ']}"
        for summary in sample_summaries
    ]
    label_map = dict(zip(ordered_ids, labels, strict=True))

    matrix = overlap.pivot(index='left', columns='right', values='jaccard').loc[ordered_ids, ordered_ids]
    matrix.index = [label_map[sample_id] for sample_id in matrix.index]
    matrix.columns = [label_map[sample_id] for sample_id in matrix.columns]

    figsize = max(6.0, 0.8 * len(sample_summaries))
    plt.figure(figsize=(figsize, figsize))
    ax = sns.heatmap(
        matrix,
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Jaccard overlap'},
    )
    ax.set_title('Pairwise gene-panel overlap')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = output_path.with_suffix('.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    assert output_path.exists(), f'Plot not written: {output_path}'
    assert png_path.exists(), f'Plot not written: {png_path}'
    return output_path


def report_feature_overlap(artifacts_cfg: ArtifactsConfig) -> tuple[str, Path]:
    managed_paths = get_managed_paths(artifacts_cfg.name)
    items_path = managed_paths.output_dir / 'items' / 'all.json'
    metadata_path = managed_paths.processed_dir / 'metadata.parquet'
    stats_path = managed_paths.output_dir / 'statistics' / 'all.parquet'
    output_path = (
        managed_paths.output_dir
        / 'figures'
        / 'items'
        / 'gene_panel_overlap'
        / f'{artifacts_cfg.items.name}.pdf'
    )

    assert items_path.exists(), f'Items not found: {items_path}'
    assert metadata_path.exists(), f'Metadata not found: {metadata_path}'
    assert stats_path.exists(), f'Statistics not found: {stats_path}'

    selected_items, missing_include_ids = select_artifact_items(
        artifacts_cfg,
        items_path=items_path,
        metadata_path=metadata_path,
        stats_path=stats_path,
    )
    items = load_items_with_metadata_from_frame(selected_items, metadata_path)
    sample_summaries = collect_sample_summaries(items)
    overlap = compute_pairwise_overlap(sample_summaries)
    plot_pairwise_overlap(sample_summaries, overlap, output_path)
    return build_overlap_report(sample_summaries, overlap, missing_include_ids), output_path
