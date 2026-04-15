"""
List failed/crashed/incomplete wandb runs and emit a bash resubmit script.

A run is selected for resubmission if it satisfies ANY of:
  - state in {failed, crashed}
  - last epoch < MIN_EPOCHS  (default 20)
  - state is not running (running jobs are always excluded)

Usage:
    uv run python scripts/train/resubmit_failed.py \
        --project xe-hne-fus-expr-v0 \
        --min_epochs 20 \
        --partition gpu-l40 \
        --memory 64G \
        --time 04:00:00 \
        --out resubmit.sh

    # Preview without writing:
    uv run python scripts/train/resubmit_failed.py --project xe-hne-fus-expr-v0 --dry_run true
"""

import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

load_dotenv(override=True)

ENTITY = "chuv"
BAD_STATES = {"failed", "crashed"}
EXCLUDE_STATES = {"running"}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, required=True, help="W&B project name.")
    parser.add_argument("--min_epochs", type=int, default=20, help="Runs with fewer epochs are resubmitted.")
    parser.add_argument("--partition", type=str, default="gpu-l40")
    parser.add_argument("--memory", type=str, default="64G")
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--out", type=Path, default=Path("resubmit.sh"), help="Output bash script path.")
    parser.add_argument("--dry_run", type=bool, default=False, help="Print sbatch commands without writing the script.")
    return parser.parse_args(argv)


def epoch_count(run) -> int | None:
    """Return the last logged epoch from run summary, or None if unavailable."""
    epoch = run.summary.get("epoch")
    if epoch is not None:
        return int(epoch)
    # trainer logs it as trainer/global_step — use as proxy only if epoch missing
    return None


def extract_run_config(run) -> dict | None:
    """
    Extract resubmission parameters from a wandb run config.

    Returns None and prints a warning if required fields are missing.
    """
    cfg = run.config

    metadata_path = cfg.get("data", {}).get("metadata_path")
    panel_path = cfg.get("data", {}).get("panel_path")
    task = cfg.get("task", {}).get("target")
    model = cfg.get("wandb", {}).get("name")
    fusion_strategy = cfg.get("backbone", {}).get("fusion_strategy")
    learnable_gate = cfg.get("backbone", {}).get("learnable_gate")

    # organ is the first non-'hest1k' tag
    tags = run.tags or []
    organ = next((t for t in tags if t != "hest1k"), None)

    missing = [k for k, v in [
        ("data.metadata_path", metadata_path),
        ("data.panel_path", panel_path),
        ("task.target", task),
        ("wandb.name", model),
        ("organ tag", organ),
    ] if v is None]

    if missing:
        print(f"  WARNING: run {run.name} ({run.id}) missing fields: {missing} — skipped", file=sys.stderr)
        return None

    # Derive outer fold from metadata_path: e.g. "breast/outer=2-inner=0-seed=0.parquet"
    outer = None
    for part in Path(metadata_path).stem.split("-"):
        if part.startswith("outer="):
            outer = part.split("=")[1]
            break

    config_path = (
        cfg.get("wandb", {}).get("config_path")
        or f"configs/train/hest1k/{task}/{organ}/{model}.yaml"
    )

    return {
        "config": config_path,
        "metadata_path": metadata_path,
        "panel_path": panel_path,
        "organ": organ,
        "task": task,
        "model": model,
        "outer": outer,
        "fusion_strategy": fusion_strategy,
        "learnable_gate": learnable_gate,
    }


def build_sbatch(r: dict, partition: str, memory: str, time: str) -> str:
    is_concat = r["fusion_strategy"] == "concat"
    job_name_parts = [r["organ"], r["task"], r["model"]]
    if is_concat:
        job_name_parts.append("concat")
    if r["outer"] is not None:
        job_name_parts.append(r["outer"])
    job_name = "-".join(job_name_parts)

    wrap_cmd = (
        f"uv run python scripts/train/supervised.py"
        f" --config {r['config']}"
        f" --data.metadata_path {r['metadata_path']}"
        f" --data.panel_path {r['panel_path']}"
    )
    if is_concat:
        wrap_cmd += " --backbone.fusion_strategy concat"
    if r["learnable_gate"] is not None:
        wrap_cmd += f" --backbone.learnable_gate {str(r['learnable_gate']).lower()}"

    return (
        f"sbatch \\\n"
        f"    --cpus-per-task=12 \\\n"
        f"    --mem={memory} \\\n"
        f"    --gres=gpu:1 \\\n"
        f"    --partition={partition} \\\n"
        f"    --time={time} \\\n"
        f"    --output=$HOME/logs/%j.out \\\n"
        f"    --job-name={job_name} \\\n"
        f"    --wrap=\"{wrap_cmd}\""
    )


def main(argv=None):
    args = parse_args(argv)

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{args.project}")

    selected = []
    skipped_running = 0

    for run in runs:
        if run.state in EXCLUDE_STATES:
            skipped_running += 1
            continue

        epochs = epoch_count(run)
        is_bad_state = run.state in BAD_STATES
        is_short = epochs is not None and epochs < args.min_epochs

        if not (is_bad_state or is_short):
            continue

        reason = []
        if is_bad_state:
            reason.append(f"state={run.state}")
        if is_short:
            reason.append(f"epochs={epochs}<{args.min_epochs}")

        print(f"  [{', '.join(reason)}] {run.name} ({run.id})")

        params = extract_run_config(run)
        if params is not None:
            selected.append(params)

    print(f"\n{skipped_running} running job(s) excluded.")
    print(f"{len(selected)} run(s) selected for resubmission.")

    if not selected:
        print("Nothing to resubmit.")
        return 0

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for r in selected:
        lines.append(build_sbatch(r, args.partition, args.memory, args.time))
        lines.append("")

    script = "\n".join(lines)

    if args.dry_run:
        print("\n--- dry run: sbatch commands ---\n")
        print(script)
        return 0

    args.out.write_text(script)
    args.out.chmod(0o755)
    print(f"\nResubmit script written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
