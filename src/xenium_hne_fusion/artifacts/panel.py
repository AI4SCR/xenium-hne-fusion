from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from xenium_hne_fusion.artifacts.items import load_items_dataframe


def intersect_gene_universes(sample_ids: list[str], processed_dir: Path) -> list[str]:
    assert sample_ids, 'No sample ids provided'
    gene_orders = [
        load_feature_universe(processed_dir / sample_id / 'feature_universe.txt') for sample_id in sample_ids
    ]

    common_genes = set(gene_orders[0])
    for gene_order in gene_orders[1:]:
        common_genes &= set(gene_order)

    canonical_order = [gene for gene in gene_orders[0] if gene in common_genes]
    assert canonical_order, f'No common genes found across samples: {sample_ids}'
    return canonical_order


def _save_panel(output_path: Path, source_panel: list[str], target_panel: list[str], overwrite: bool = False) -> Path:
    if output_path.exists():
        assert overwrite, f'Panel already exists: {output_path}'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            {'source_panel': source_panel, 'target_panel': target_panel},
            sort_keys=False,
        )
    )
    logger.info(f'Saved panel ({len(source_panel)} source, {len(target_panel)} target genes) -> {output_path}')
    return output_path


def save_source_panel(output_path: Path, source_panel: list[str], overwrite: bool = False) -> Path:
    return _save_panel(output_path, source_panel, [], overwrite=overwrite)


def build_panel(items_path: Path, processed_dir: Path, output_path: Path, overwrite: bool = False) -> Path:
    sample_ids = load_items_dataframe(items_path)['sample_id'].unique().tolist()
    source_panel = intersect_gene_universes(sample_ids, processed_dir)
    logger.info(f'Found {len(source_panel)} common genes across {len(sample_ids)} samples')
    return save_source_panel(output_path, source_panel, overwrite=overwrite)


def load_transcript_gene_categories(transcripts_path: Path) -> list[str]:
    transcripts = pd.read_parquet(transcripts_path, columns=['feature_name'])
    feature_name = transcripts['feature_name']
    assert isinstance(feature_name.dtype, pd.CategoricalDtype), (
        f'Expected categorical feature_name in {transcripts_path}, got {feature_name.dtype}'
    )
    return feature_name.cat.categories.tolist()


def load_feature_universe(feature_universe_path: Path) -> list[str]:
    assert feature_universe_path.exists(), f'Missing feature universe: {feature_universe_path}'
    genes = feature_universe_path.read_text().splitlines()
    assert genes, f'Empty feature universe: {feature_universe_path}'
    return genes
