import json
from pathlib import Path

from xenium_hne_fusion.artifacts.items import create_items


def test_create_items_collects_all_tiles_with_tile_pt(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    processed_dir = tmp_path / '02_processed' / 'hest1k'

    with_transcripts = processed_dir / 'TENX95' / '256_256' / '0'
    without_transcripts = processed_dir / 'TENX95' / '256_256' / '1'
    for tile_dir in [with_transcripts, without_transcripts]:
        tile_dir.mkdir(parents=True, exist_ok=True)
        (tile_dir / 'tile.pt').write_text('')
    (with_transcripts / 'expr-kernel_size=16.parquet').write_text('')
    (with_transcripts / 'transcripts.parquet').write_text('')

    items_path = create_items(output_dir / 'items', processed_dir, tile_px=256, stride_px=256, overwrite=True)

    assert items_path == output_dir / 'items' / 'all.json'
    items = json.loads(items_path.read_text())
    assert items == [
        {
            'id': 'TENX95_0',
            'sample_id': 'TENX95',
            'tile_id': 0,
            'tile_dir': str(with_transcripts),
        },
        {
            'id': 'TENX95_1',
            'sample_id': 'TENX95',
            'tile_id': 1,
            'tile_dir': str(without_transcripts),
        },
    ]


def test_create_items_skips_when_already_exists(tmp_path: Path):
    output_dir = tmp_path / '03_output' / 'hest1k'
    processed_dir = tmp_path / '02_processed' / 'hest1k'
    items_path = output_dir / 'items' / 'all.json'
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps([{'id': 'existing'}]))

    result = create_items(output_dir / 'items', processed_dir, tile_px=256, stride_px=256, overwrite=False)

    assert result == items_path
    assert json.loads(items_path.read_text()) == [{'id': 'existing'}]
