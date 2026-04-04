"""Save WSI thumbnail and transcript overview for quick inspection."""
from __future__ import annotations

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import save_transcript_overview, save_wsi_thumbnail


def main(
    wsi_path: Path,
    transcripts_path: Path,
    output_dir: Path,
    n: int = 10_000,
    max_size: int = 2048,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_wsi_thumbnail(wsi_path, output_dir / "wsi.png", max_size=max_size)
    save_transcript_overview(
        wsi_path,
        transcripts_path,
        output_dir / "transcripts.png",
        n=n,
        max_size=max_size,
    )


if __name__ == "__main__":
    auto_cli(main)
