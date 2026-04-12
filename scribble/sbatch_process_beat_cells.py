"""Submit one Slurm job per BEAT sample for cell annotation processing."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.processing import process_cells, tile_cells
from xenium_hne_fusion.utils.getters import (
    PipelineConfig,
    build_pipeline_config,
    load_processing_config,
    processed_sample_dir,
    select_sample_ids,
)


def resolve_beat_samples(cfg: PipelineConfig) -> list[str]:
    sample_ids = sorted(path.name for path in cfg.paths.structured_dir.iterdir() if path.is_dir())
    return select_sample_ids(sample_ids, cfg.data.filter)


def process_sample_cells(cfg: PipelineConfig, sample_id: str) -> None:
    tiles_cfg = cfg.data.tiles
    assert tiles_cfg.img_size is not None, "tiles.img_size is required"

    structured_dir = cfg.paths.structured_dir / sample_id
    cells_path = structured_dir / "cells.parquet"
    tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
    output_dir = processed_sample_dir(cfg, sample_id)

    if not cells_path.exists():
        logger.warning(f"Skipping {sample_id}: missing cells.parquet")
        return

    assert tiles_path.exists(), f"Missing tiles: {tiles_path}"
    assert output_dir.exists(), f"Missing processed sample dir: {output_dir}"

    tiles = gpd.read_parquet(tiles_path)
    tile_cells(tiles, cells_path, output_dir, predicate=tiles_cfg.predicate)
    process_cells(tiles, output_dir, img_size=tiles_cfg.img_size)


def submit_sample_job(
    *,
    config_path: Path,
    sample_id: str,
    cpus: int,
    mem: str,
    time: str,
    log_dir: Path,
    dry_run: bool,
) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        "uv",
        "run",
        "python",
        "scribble/sbatch_process_beat_cells.py",
        "--config",
        str(config_path),
        "--sample-id",
        sample_id,
        "--worker",
    ]
    wrapped_cmd = f"cd {shlex.quote(str(repo_root))} && {shlex.join(cmd)}"
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"beat_cells_{sample_id}",
        "--cpus-per-task",
        str(cpus),
        "--mem",
        mem,
        "--time",
        time,
        "--output",
        str(log_dir / "%j.log"),
        "--error",
        str(log_dir / "%j.err"),
        "--wrap",
        wrapped_cmd,
    ]

    if dry_run:
        print(shlex.join(sbatch_cmd))
        return None

    result = subprocess.run(sbatch_cmd, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    logger.info(f"Submitted {sample_id}: {job_id}")
    return job_id


def main(
    *,
    config_path: Path,
    sample_id: str | None,
    worker: bool,
    cpus: int,
    mem: str,
    time: str,
    log_dir: Path,
    dry_run: bool,
) -> None:
    load_dotenv()
    config_path = config_path.resolve()
    data_cfg = load_processing_config(config_path)
    assert data_cfg.name == "beat", f"Expected dataset='beat', got {data_cfg.name!r}"

    cfg = build_pipeline_config(data_cfg)
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if worker:
        assert sample_id is not None, "--sample-id is required in worker mode"
        process_sample_cells(cfg, sample_id)
        return

    assert sample_id is None, "--sample-id is only valid with --worker"
    sample_ids = resolve_beat_samples(cfg)
    assert sample_ids, f"No structured BEAT samples found in {cfg.paths.structured_dir}"
    logger.info(f"Submitting BEAT cell jobs for {len(sample_ids)} samples")

    job_ids = [
        submit_sample_job(
            config_path=config_path,
            sample_id=current_sample_id,
            cpus=cpus,
            mem=mem,
            time=time,
            log_dir=log_dir,
            dry_run=dry_run,
        )
        for current_sample_id in sample_ids
    ]
    if not dry_run:
        print("\n".join(job_id for job_id in job_ids if job_id is not None))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-id", type=str)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="08:00:00")
    parser.add_argument("--log-dir", type=Path, default=Path("~/logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    main(
        config_path=args.config,
        sample_id=args.sample_id,
        worker=args.worker,
        cpus=args.cpus,
        mem=args.mem,
        time=args.time,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
