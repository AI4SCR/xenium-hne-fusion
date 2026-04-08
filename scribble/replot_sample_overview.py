"""Regenerate transcript overview overlays for a structured sample."""

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import save_transcript_overview


def main(
    sample_dir: Path,
    output_path: Path | None = None,
    n: int = 10_000,
    max_size: int = 2048,
    seed: int = 0,
) -> None:
    sample_dir = Path(sample_dir)
    wsi_path = sample_dir / "wsi.tiff"
    transcripts_path = sample_dir / "transcripts.parquet"
    output_path = output_path or sample_dir / "transcripts.png"

    assert wsi_path.exists(), f"Missing WSI: {wsi_path}"
    assert transcripts_path.exists(), f"Missing transcripts: {transcripts_path}"

    save_transcript_overview(
        wsi_path=wsi_path,
        transcripts_path=transcripts_path,
        output_path=output_path,
        n=n,
        max_size=max_size,
        seed=seed,
    )


if __name__ == "__main__":
    auto_cli(main)
