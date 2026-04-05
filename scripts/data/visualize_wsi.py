"""Save WSI thumbnail and transcript overview for quick inspection."""

from pathlib import Path

from jsonargparse import auto_cli

from xenium_hne_fusion.tiling import save_sample_overview


def main(
    wsi_path: Path,
    transcripts_path: Path,
    output_dir: Path,
    n: int = 10_000,
    max_size: int = 2048,
) -> None:
    save_sample_overview(wsi_path, transcripts_path, output_dir, n=n, max_size=max_size)


if __name__ == "__main__":
    auto_cli(main)
