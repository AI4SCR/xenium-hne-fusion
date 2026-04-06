import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _load_script(path: str, module_name: str):
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_expr(tile_dir: Path, genes: list[str], *, include_token_index: bool = False) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    data = {gene: [1, 0] for gene in genes}
    if include_token_index:
        data = {'token_index': [0, 1], **data}
    pd.DataFrame(data).to_parquet(tile_dir / 'expr-kernel_size=16.parquet', index=False)


def test_report_overlap_script_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    module = _load_script('.codex/skills/hest1k-panel-overlap/scripts/report_overlap.py', 'report_overlap_script')

    s1_tile = tmp_path / 'S1' / '0'
    s2_tile = tmp_path / 'S2' / '0'
    _write_expr(s1_tile, ['A', 'B', 'C'])
    _write_expr(s2_tile, ['A', 'D'], include_token_index=True)

    items_path = tmp_path / 'items.json'
    items_path.write_text(
        json.dumps(
            [
                {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(s1_tile)},
                {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': str(s2_tile)},
            ]
        )
    )
    metadata_path = tmp_path / 'metadata.parquet'
    pd.DataFrame(
        [
            {'sample_id': 'S1', 'organ': 'Breast'},
            {'sample_id': 'S2', 'organ': 'Lung'},
        ]
    ).to_parquet(metadata_path, index=False)
    panel_path = tmp_path / 'panel.yaml'
    panel_path.write_text(yaml.safe_dump({'source_panel': ['A'], 'target_panel': ['C']}, sort_keys=False))

    assert module.main(
        ['--items-path', str(items_path), '--metadata-path', str(metadata_path), '--panel-path', str(panel_path)]
    ) == 0

    output = capsys.readouterr().out
    assert 'Sample summaries' in output
    assert '- S1: organ=Breast genes=3' in output
    assert '- S2: organ=Lung genes=2' in output
    assert '- Breast vs Lung: intersection=1 union=4 jaccard=0.250' in output
    assert '- S2: present=1 missing=1' in output


def test_hest1k_organ_panels_match_selected_items_when_data_available():
    module = _load_script('.codex/skills/hest1k-panel-overlap/scripts/report_overlap.py', 'report_overlap_real_data')

    metadata_path = Path('data/02_processed/hest1k/metadata.parquet')
    if not metadata_path.exists():
        pytest.skip('Processed hest1k metadata not available')

    for organ in ['breast', 'lung', 'pancreas']:
        items_path = Path(f'data/03_output/hest1k/items/{organ}.json')
        panel_path = Path(f'panels/hest1k/hvg-{organ}-default-outer=0-seed=0.yaml')
        if not items_path.exists() or not panel_path.exists():
            pytest.skip(f'Missing data for {organ}')

        items = module.load_items_with_metadata(items_path, metadata_path)
        summaries = module.collect_sample_summaries(items)
        panel_genes = module.load_panel_genes(panel_path)
        compatibility = module.compute_panel_compatibility(
            panel_genes,
            [(summary['sample_id'], summary['genes']) for summary in summaries],
        )

        assert compatibility
        assert all(row['missing'] == 0 for row in compatibility)
