from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

TOKEN_INDEX_COLUMN = 'token_index'


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_items_with_metadata(items_path: Path, metadata_path: Path) -> pd.DataFrame:
    items = pd.DataFrame(json.loads(items_path.read_text()))
    metadata = pd.read_parquet(metadata_path)
    if 'sample_id' not in metadata.columns and 'id' in metadata.columns:
        metadata = metadata.rename(columns={'id': 'sample_id'})
    organ_cols = [col for col in ['organ'] if col in metadata.columns]
    if organ_cols:
        metadata = metadata[['sample_id', *organ_cols]].drop_duplicates()
        items = items.merge(metadata, on='sample_id', how='left')
    return items


def load_expr_genes(expr_path: Path) -> tuple[str, ...]:
    names = pq.ParquetFile(expr_path).schema_arrow.names
    return tuple(name for name in names if name != TOKEN_INDEX_COLUMN)


def collect_sample_summaries(items: pd.DataFrame) -> list[dict]:
    summaries: list[dict] = []
    for sample_id, group in items.sort_values(['sample_id', 'tile_id']).groupby('sample_id', sort=True):
        signatures: dict[tuple[str, ...], int] = {}
        for tile_dir in group['tile_dir']:
            genes = load_expr_genes(Path(tile_dir) / 'expr-kernel_size=16.parquet')
            signatures[genes] = signatures.get(genes, 0) + 1

        ordered = sorted(signatures.items(), key=lambda item: (-item[1], item[0]))
        canonical_genes = ordered[0][0]
        organ = group['organ'].iloc[0] if 'organ' in group.columns else None
        summaries.append(
            {
                'sample_id': sample_id,
                'organ': organ,
                'num_tiles': len(group),
                'num_signatures': len(signatures),
                'signature_sizes': sorted({len(genes) for genes in signatures}),
                'genes': canonical_genes,
            }
        )
    return summaries


def collect_organ_summaries(sample_summaries: list[dict]) -> list[dict]:
    by_organ: dict[str, list[tuple[str, ...]]] = {}
    for summary in sample_summaries:
        organ = summary['organ']
        if organ is None:
            continue
        by_organ.setdefault(organ, []).append(summary['genes'])

    organ_summaries: list[dict] = []
    for organ in sorted(by_organ):
        gene_sets = [set(genes) for genes in by_organ[organ]]
        shared = tuple(sorted(set.intersection(*gene_sets))) if gene_sets else tuple()
        organ_summaries.append(
            {
                'organ': organ,
                'num_samples': len(gene_sets),
                'genes': shared,
            }
        )
    return organ_summaries


def compute_pairwise_overlap(named_gene_sets: list[tuple[str, tuple[str, ...]]]) -> list[dict]:
    rows: list[dict] = []
    for (name_a, genes_a), (name_b, genes_b) in itertools.combinations(named_gene_sets, 2):
        set_a = set(genes_a)
        set_b = set(genes_b)
        inter = set_a & set_b
        union = set_a | set_b
        rows.append(
            {
                'left': name_a,
                'right': name_b,
                'intersection': len(inter),
                'union': len(union),
                'jaccard': len(inter) / len(union) if union else 1.0,
            }
        )
    return rows


def load_panel_genes(panel_path: Path) -> tuple[str, ...]:
    panel = yaml.safe_load(panel_path.read_text()) or {}
    return tuple(panel.get('source_panel', [])) + tuple(panel.get('target_panel', []))


def compute_panel_compatibility(panel_genes: tuple[str, ...], named_gene_sets: list[tuple[str, tuple[str, ...]]]) -> list[dict]:
    rows: list[dict] = []
    for name, genes in named_gene_sets:
        gene_set = set(genes)
        missing = [gene for gene in panel_genes if gene not in gene_set]
        rows.append(
            {
                'name': name,
                'present': len(panel_genes) - len(missing),
                'missing': len(missing),
                'missing_genes': missing,
            }
        )
    return rows


def build_report(items: pd.DataFrame, panel_path: Path | None = None) -> str:
    sample_summaries = collect_sample_summaries(items)
    organ_summaries = collect_organ_summaries(sample_summaries)

    lines = ['Sample summaries']
    for summary in sample_summaries:
        lines.append(
            f"- {summary['sample_id']}: organ={summary['organ']} genes={len(summary['genes'])} "
            f"tiles={summary['num_tiles']} signatures={summary['num_signatures']} "
            f"signature_sizes={summary['signature_sizes']}"
        )

    sample_pairs = compute_pairwise_overlap([(row['sample_id'], row['genes']) for row in sample_summaries])
    if sample_pairs:
        lines.append('')
        lines.append('Sample pairwise overlap')
        for row in sample_pairs:
            lines.append(
                f"- {row['left']} vs {row['right']}: intersection={row['intersection']} "
                f"union={row['union']} jaccard={row['jaccard']:.3f}"
            )

    if organ_summaries:
        lines.append('')
        lines.append('Organ summaries')
        for summary in organ_summaries:
            lines.append(f"- {summary['organ']}: samples={summary['num_samples']} shared_genes={len(summary['genes'])}")

        organ_pairs = compute_pairwise_overlap([(row['organ'], row['genes']) for row in organ_summaries])
        if organ_pairs:
            lines.append('')
            lines.append('Organ pairwise overlap')
            for row in organ_pairs:
                lines.append(
                    f"- {row['left']} vs {row['right']}: intersection={row['intersection']} "
                    f"union={row['union']} jaccard={row['jaccard']:.3f}"
                )

        all_sets = [set(row['genes']) for row in organ_summaries]
        if all_sets:
            intersection = set.intersection(*all_sets)
            union = set.union(*all_sets)
            lines.append('')
            lines.append(
                f"All organs: intersection={len(intersection)} union={len(union)} "
                f"jaccard={len(intersection) / len(union) if union else 1.0:.3f}"
            )

    if panel_path is not None:
        panel_genes = load_panel_genes(panel_path)
        lines.append('')
        lines.append(f'Panel compatibility: {panel_path}')
        for row in compute_panel_compatibility(panel_genes, [(r['sample_id'], r['genes']) for r in sample_summaries]):
            lines.append(
                f"- {row['name']}: present={row['present']} missing={row['missing']} "
                f"missing_preview={row['missing_genes'][:8]}"
            )

    return '\n'.join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description='Report HEST1K sample and organ gene overlap.')
    parser.add_argument('--items-path', type=Path, default=repo_root / 'data/03_output/hest1k/items/all.json')
    parser.add_argument('--metadata-path', type=Path, default=repo_root / 'data/02_processed/hest1k/metadata.parquet')
    parser.add_argument('--panel-path', type=Path, default=None)
    parser.add_argument('--organ', action='append', default=None)
    parser.add_argument('--sample-id', action='append', default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = load_items_with_metadata(args.items_path, args.metadata_path)
    if args.organ is not None and 'organ' in items.columns:
        items = items[items['organ'].isin(args.organ)]
    if args.sample_id is not None:
        items = items[items['sample_id'].isin(args.sample_id)]

    assert len(items) > 0, 'No items selected'
    print(build_report(items, panel_path=args.panel_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
