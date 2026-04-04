from pathlib import Path

from xenium_hne_fusion.download import create_structured_symlinks


def test_create_structured_symlinks_creates_sample_visualizations(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / 'raw'
    structured_dir = tmp_path / 'structured'
    wsi_path = raw_dir / 'wsis' / 'TENX95.tif'
    transcripts_path = raw_dir / 'transcripts' / 'TENX95_transcripts.parquet'

    wsi_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)
    wsi_path.write_text('wsi')
    transcripts_path.write_text('transcripts')

    calls = []

    def fake_save_sample_overview(wsi_path: Path, transcripts_path: Path, output_dir: Path, n: int = 10_000, max_size: int = 2048):
        calls.append((wsi_path, transcripts_path, output_dir, n, max_size))

    monkeypatch.setattr('xenium_hne_fusion.download.save_sample_overview', fake_save_sample_overview)

    create_structured_symlinks('TENX95', raw_dir, structured_dir)

    sample_dir = structured_dir / 'TENX95'
    assert (sample_dir / 'wsi.tiff').is_symlink()
    assert (sample_dir / 'transcripts.parquet').is_symlink()
    assert calls == [(sample_dir / 'wsi.tiff', sample_dir / 'transcripts.parquet', sample_dir, 10_000, 2048)]
