from pathlib import Path

import pandas as pd

from xenium_hne_fusion.panel_overlap import build_overlap_title, collect_sample_summaries


def _write_feature_universe(tile_dir: Path, genes: list[str]) -> None:
    sample_dir = tile_dir.parent.parent
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / 'feature_universe.txt').write_text('\n'.join(genes) + '\n')


def test_collect_sample_summaries_uses_sample_feature_universe(tmp_path: Path):
    s1_tile0 = tmp_path / 'S1' / '256_256' / '0'
    s1_tile1 = tmp_path / 'S1' / '256_256' / '1'
    s2_tile0 = tmp_path / 'S2' / '256_256' / '0'
    s1_tile0.mkdir(parents=True, exist_ok=True)
    s1_tile1.mkdir(parents=True, exist_ok=True)
    s2_tile0.mkdir(parents=True, exist_ok=True)
    _write_feature_universe(s1_tile0, ['A', 'B', 'C'])
    _write_feature_universe(s2_tile0, ['A', 'C', 'D'])

    items = pd.DataFrame(
        [
            {'id': 'S2_0', 'sample_id': 'S2', 'tile_id': 0, 'tile_dir': str(s2_tile0), 'organ': 'Lung'},
            {'id': 'S1_1', 'sample_id': 'S1', 'tile_id': 1, 'tile_dir': str(s1_tile1), 'organ': 'Breast'},
            {'id': 'S1_0', 'sample_id': 'S1', 'tile_id': 0, 'tile_dir': str(s1_tile0), 'organ': 'Breast'},
        ]
    )

    summaries = collect_sample_summaries(items)

    assert summaries == [
        {
            'sample_id': 'S1',
            'organ': 'Breast',
            'num_tiles': 2,
            'genes': ('A', 'B', 'C'),
        },
        {
            'sample_id': 'S2',
            'organ': 'Lung',
            'num_tiles': 1,
            'genes': ('A', 'C', 'D'),
        },
    ]


def test_build_overlap_title_reports_pairwise_and_global_intersections():
    sample_summaries = [
        {'sample_id': 'S1', 'organ': 'Breast', 'num_tiles': 1, 'genes': ('A', 'B', 'C', 'D')},
        {'sample_id': 'S2', 'organ': 'Lung', 'num_tiles': 1, 'genes': ('B', 'C', 'D', 'E')},
        {'sample_id': 'S3', 'organ': 'Pancreas', 'num_tiles': 1, 'genes': ('C', 'D', 'F')},
    ]
    overlap_rows = [
        {'left': 'S1', 'right': 'S1', 'intersection': 4, 'union': 4, 'jaccard': 1.0},
        {'left': 'S1', 'right': 'S2', 'intersection': 3, 'union': 5, 'jaccard': 0.6},
        {'left': 'S1', 'right': 'S3', 'intersection': 2, 'union': 5, 'jaccard': 0.4},
        {'left': 'S2', 'right': 'S1', 'intersection': 3, 'union': 5, 'jaccard': 0.6},
        {'left': 'S2', 'right': 'S2', 'intersection': 4, 'union': 4, 'jaccard': 1.0},
        {'left': 'S2', 'right': 'S3', 'intersection': 2, 'union': 5, 'jaccard': 0.4},
        {'left': 'S3', 'right': 'S1', 'intersection': 2, 'union': 5, 'jaccard': 0.4},
        {'left': 'S3', 'right': 'S2', 'intersection': 2, 'union': 5, 'jaccard': 0.4},
        {'left': 'S3', 'right': 'S3', 'intersection': 3, 'union': 3, 'jaccard': 1.0},
    ]

    title = build_overlap_title(sample_summaries, pd.DataFrame(overlap_rows))

    assert title == (
        'Pairwise gene-panel overlap\n'
        'pairwise intersection min=2 max=3 all-sample intersection=2'
    )


