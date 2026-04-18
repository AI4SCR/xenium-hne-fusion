"""
Query W&B for failed/crashed/incomplete runs, group duplicate experiment definitions,
and emit a bash resubmit script for groups that have no successful run.

A group is solved if it contains at least one run with state == "finished" and epoch >= MIN_EPOCHS.
A group is resubmitted if it has no successful run and at least one failed/crashed/incomplete candidate.

Deletion policy (--delete true) is conservative:
  - only groups with >1 runs are considered
  - only groups with at least one successful run are considered
  - keep the latest successful run
  - keep one failure exemplar (prefer failed/crashed)
  - never delete running runs

Usage:
    uv run python scripts/train/resubmit.py --project xe-hne-fus-expr-v0

    # Preview sbatch commands without writing:
    uv run python scripts/train/resubmit.py --project xe-hne-fus-expr-v0 --dry true

    # Write per-run and per-group tables:
    uv run python scripts/train/resubmit.py \\
        --project xe-hne-fus-expr-v0 \\
        --table_out run_table.csv \\
        --group_table_out group_table.csv

    # Preview deletion plan:
    uv run python scripts/train/resubmit.py --project xe-hne-fus-expr-v0 --dry true --delete true

    # Execute deletions:
    uv run python scripts/train/resubmit.py --project xe-hne-fus-expr-v0 --delete true
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

load_dotenv(override=True)

ENTITY = "chuv"
BAD_STATES = {"failed", "crashed"}
EXCLUDE_STATES = {"running"}
DATASET_TAGS = {"hest1k", "hescape", "beat"}
KNOWN_ORGANS = {"lung", "lung-healthy", "breast", "bowel", "pancreas", "human-immuno-oncology", "human-multi-tissue"}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, required=True, help="W&B project name.")
    parser.add_argument("--min_epochs", type=int, default=20, help="Runs with fewer epochs are treated as incomplete.")
    parser.add_argument("--partition", type=str, default="gpu-l40")
    parser.add_argument("--memory", type=str, default="64G")
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--out", type=Path, default=Path("resubmit.sh"), help="Output bash script path.")
    parser.add_argument("--table_out", type=Path, default=None, help="Optional CSV with one row per run.")
    parser.add_argument("--group_table_out", type=Path, default=None, help="Optional CSV with one row per group.")
    parser.add_argument("--dry", type=bool, default=True, help="Print sbatch commands / deletion plan without writing.")
    parser.add_argument("--delete", type=bool, default=False, help="Delete duplicate runs and local log dirs conservatively.")
    parser.add_argument("--delete_plan_out", type=Path, default=Path("delete_plan.csv"), help="CSV of runs marked for deletion; written on dry runs for inspection.")
    parser.add_argument(
        "--log_dir_root",
        type=Path,
        default=Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/logs/xe-hne-fus-expr-v0"),
        help="Root directory containing per-run log directories named by run.id.",
    )
    return parser.parse_args(argv)


def epoch_count(run) -> int | None:
    """Return the last logged epoch from run summary, or None if unavailable."""
    epoch = run.summary.get("epoch")
    if epoch is not None:
        return int(epoch)
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
    dataset_name = cfg.get("data", {}).get("name")
    fusion_strategy = cfg.get("backbone", {}).get("fusion_strategy")
    fusion_stage = cfg.get("backbone", {}).get("fusion_stage")
    learnable_gate = cfg.get("backbone", {}).get("learnable_gate")
    freeze_morph_encoder = cfg.get("backbone", {}).get("freeze_morph_encoder")
    freeze_expr_encoder = cfg.get("backbone", {}).get("freeze_expr_encoder")

    tags = run.tags or []
    organ = next((t for t in tags if t not in DATASET_TAGS and t in KNOWN_ORGANS), None)
    if organ is None and dataset_name == "beat":
        organ = "lung"

    required = {
        "data.metadata_path": metadata_path,
        "data.panel_path": panel_path,
        "data.name": dataset_name,
        "task.target": task,
        "wandb.name": model,
        "organ tag": organ,
    }
    missing = [k for k, v in required.items() if v is None]

    if missing:
        print(f"  WARNING: run {run.name} ({run.id}) missing fields: {missing} — skipped", file=sys.stderr)
        return None

    outer = None
    for part in Path(metadata_path).stem.split("-"):
        if part.startswith("outer="):
            outer = part.split("=")[1]
            break

    config_path = cfg.get("wandb", {}).get("config_path")
    if config_path is None:
        if dataset_name in {"hest1k", "hescape"}:
            config_path = f"configs/train/{dataset_name}/{task}/{organ}/{model}.yaml"
        else:
            config_path = f"configs/train/{dataset_name}/{task}/{model}.yaml"

    return {
        "config": config_path,
        "metadata_path": metadata_path,
        "panel_path": panel_path,
        "dataset_name": dataset_name,
        "organ": organ,
        "task": task,
        "model": model,
        "outer": outer,
        "fusion_stage": fusion_stage,
        "fusion_strategy": fusion_strategy,
        "learnable_gate": learnable_gate,
        "freeze_morph_encoder": freeze_morph_encoder,
        "freeze_expr_encoder": freeze_expr_encoder,
    }


def make_group_key(params: dict) -> tuple[str, str]:
    """
    Define experiment identity for duplicate detection.

    Matches on the minimal set of fields that uniquely identify an experiment:
    organ, task, model, outer fold, fusion stage, fusion strategy, and
    learnable gate. All are extracted from the W&B run config/tags.
    """
    payload = {
        "dataset_name": params["dataset_name"],
        "organ": params["organ"],
        "task": params["task"],
        "model": params["model"],
        "panel_path": params["panel_path"],
        "outer": params["outer"],
        "fusion_stage": params["fusion_stage"],
        "fusion_strategy": params["fusion_strategy"],
        "learnable_gate": params["learnable_gate"],
        "freeze_morph_encoder": params["freeze_morph_encoder"],
        "freeze_expr_encoder": params["freeze_expr_encoder"],
    }
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return key, key_hash


def build_run_record(run, min_epochs: int) -> dict | None:
    params = extract_run_config(run)
    if params is None:
        return None

    epochs = epoch_count(run)
    is_bad_state = run.state in BAD_STATES
    is_short = epochs is not None and epochs < min_epochs
    is_success = run.state == "finished" and epochs is not None and epochs >= min_epochs
    is_candidate = run.state not in EXCLUDE_STATES and (is_bad_state or is_short)

    group_key, group_hash = make_group_key(params)

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "epoch": epochs,
        "is_bad_state": is_bad_state,
        "is_short": is_short,
        "is_success": is_success,
        "is_candidate": is_candidate,
        "group_key": group_key,
        "group_hash": group_hash,
        "_run_obj": run,
        **params,
    }


def choose_candidate(records: list[dict]) -> dict | None:
    """
    Choose one representative run from an unsolved group.

    Preference:
      1. failed/crashed
      2. other incomplete runs
      3. highest epoch among candidates
    """
    candidates = [r for r in records if r["is_candidate"]]
    if not candidates:
        return None

    def key(r: dict):
        bad_state_rank = 0 if r["is_bad_state"] else 1
        epoch_rank = -(r["epoch"] if r["epoch"] is not None else -1)
        return (bad_state_rank, epoch_rank, r["run_id"])

    return sorted(candidates, key=key)[0]


def build_sbatch(r: dict, partition: str, memory: str, time: str) -> str:
    is_concat = r["fusion_strategy"] == "concat"
    job_name_parts = [r["dataset_name"], r["organ"], r["task"], r["model"]]
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    fieldnames = list(clean_rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def parse_created_at(run) -> datetime:
    value = getattr(run, "created_at", None)
    if value is None:
        return datetime.min
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def choose_latest_success(records: list[dict]) -> dict | None:
    successes = [r for r in records if r["is_success"]]
    if not successes:
        return None
    return max(successes, key=lambda r: parse_created_at(r["_run_obj"]))


def choose_latest_failure(records: list[dict]) -> dict | None:
    failed = [r for r in records if r["state"] in BAD_STATES and r["state"] not in EXCLUDE_STATES]
    if failed:
        return max(failed, key=lambda r: parse_created_at(r["_run_obj"]))
    non_success = [r for r in records if not r["is_success"] and r["state"] not in EXCLUDE_STATES]
    if non_success:
        return max(non_success, key=lambda r: parse_created_at(r["_run_obj"]))
    return None


def plan_group_deletions(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (keep, delete).

    Keep exactly one run per group:
      - if a successful run exists: keep the latest successful run
      - otherwise: keep the latest failed/incomplete run
    Running runs are never deleted. Single-run groups are left untouched.
    """
    if len(records) <= 1:
        return records[:], []

    latest_success = choose_latest_success(records)
    keeper = latest_success if latest_success is not None else choose_latest_failure(records)
    if keeper is None:
        return records[:], []

    keep_ids = {keeper["run_id"]}
    keep = [r for r in records if r["run_id"] in keep_ids or r["state"] in EXCLUDE_STATES]
    delete = [r for r in records if r["run_id"] not in keep_ids and r["state"] not in EXCLUDE_STATES]
    return keep, delete


def delete_local_log_dir(log_dir_root: Path, run_id: str) -> None:
    run_dir = log_dir_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"    deleted local logs: {run_dir}")
    else:
        print(f"    local logs not found: {run_dir}")


def execute_deletions(to_delete: list[dict], log_dir_root: Path, dry: bool) -> None:
    if not to_delete:
        return

    print("\nDeletion plan:")
    for r in to_delete:
        print(f"  DELETE {r['organ']}/{r['task']}/{r['model']} -> {r['run_name']} ({r['run_id']}) state={r['state']}")

    if dry:
        print("\nDry run enabled: no W&B runs or local log dirs were deleted.")
        return

    for r in to_delete:
        run = r["_run_obj"]
        print(f"\nDeleting run {r['run_name']} ({r['run_id']})")
        run.delete(delete_artifacts=False)
        delete_local_log_dir(log_dir_root, r["run_id"])


def main(argv=None):
    args = parse_args(argv)

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{args.project}")

    run_records = []
    skipped_running = 0

    for run in runs:
        if run.state in EXCLUDE_STATES:
            skipped_running += 1

        run.load_full_data()
        record = build_run_record(run, min_epochs=args.min_epochs)
        if record is not None:
            run_records.append(record)

    if args.table_out is not None:
        write_csv(args.table_out, run_records)
        print(f"Wrote per-run table to: {args.table_out}")

    groups = defaultdict(list)
    for r in run_records:
        groups[r["group_key"]].append(r)

    group_rows = []
    selected = []
    delete_plan = []

    print("\nGroup summary:")
    for _, records in groups.items():
        example = records[0]
        has_success = any(r["is_success"] for r in records)
        candidate = choose_candidate(records)

        success_runs = [r for r in records if r["is_success"]]
        failed_runs = [r for r in records if r["is_bad_state"]]
        short_runs = [r for r in records if r["is_short"]]

        label = f"{example['dataset_name']}/{example['organ']}/{example['task']}/{example['model']}/outer={example['outer']}"

        group_row = {
            "group_hash": example["group_hash"],
            "dataset_name": example["dataset_name"],
            "organ": example["organ"],
            "task": example["task"],
            "model": example["model"],
            "outer": example["outer"],
            "config": example["config"],
            "metadata_path": example["metadata_path"],
            "panel_path": example["panel_path"],
            "fusion_strategy": example["fusion_strategy"],
            "learnable_gate": example["learnable_gate"],
            "n_runs": len(records),
            "n_success": len(success_runs),
            "n_failed_or_crashed": len(failed_runs),
            "n_short": len(short_runs),
            "has_success": has_success,
            "selected_for_resubmit": candidate is not None and not has_success,
            "selected_run_id": None if candidate is None or has_success else candidate["run_id"],
            "selected_run_name": None if candidate is None or has_success else candidate["run_name"],
        }
        group_rows.append(group_row)

        if has_success:
            desc = ", ".join(f"{r['run_name']}({r['run_id']}, epoch={r['epoch']})" for r in success_runs)
            print(f"  [SOLVED] {label} -> {desc}")
        elif candidate is None:
            print(f"  [UNSOLVED, NO CANDIDATE] {label}")
        else:
            reasons = []
            if candidate["is_bad_state"]:
                reasons.append(f"state={candidate['state']}")
            if candidate["is_short"]:
                reasons.append(f"epochs={candidate['epoch']}<{args.min_epochs}")

            print(
                f"  [RESUBMIT] {label} from {candidate['run_name']} "
                f"({candidate['run_id']}) [{', '.join(reasons)}]"
            )
            selected.append(candidate)

        keep, to_delete = plan_group_deletions(records)
        if to_delete:
            kept_desc = ", ".join(
                f"{r['run_name']}({r['run_id']}, state={r['state']})"
                for r in keep if r["state"] != "running"
            )
            del_desc = ", ".join(
                f"{r['run_name']}({r['run_id']}, state={r['state']})"
                for r in to_delete
            )
            print(f"  [DELETE-PLAN] keep: {kept_desc}")
            print(f"  [DELETE-PLAN] delete: {del_desc}")
            delete_plan.extend(to_delete)

    if args.group_table_out is not None:
        write_csv(args.group_table_out, group_rows)
        print(f"Wrote per-group table to: {args.group_table_out}")

    solved_groups = sum(1 for row in group_rows if row["has_success"])
    unsolved_groups = len(group_rows) - solved_groups

    print(f"\n{skipped_running} running job(s) excluded from candidacy.")
    print(f"{len(run_records)} total run record(s) parsed.")
    print(f"{len(group_rows)} unique experiment group(s).")
    print(f"{solved_groups} solved group(s).")
    print(f"{unsolved_groups} unsolved group(s).")
    print(f"{len(selected)} group(s) selected for resubmission.")
    print(f"{len(delete_plan)} run(s) marked for deletion.")

    if selected:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for r in selected:
            lines.append(build_sbatch(r, args.partition, args.memory, args.time))
            lines.append("")

        script = "\n".join(lines)

        if args.dry:
            print("\n--- dry run: sbatch commands ---\n")
            print(script)
        else:
            args.out.write_text(script)
            args.out.chmod(0o755)
            print(f"\nResubmit script written to: {args.out}")
    else:
        print("Nothing to resubmit.")

    if delete_plan and args.dry:
        write_csv(args.delete_plan_out, delete_plan)
        print(f"\nDeletion plan written to: {args.delete_plan_out}")

    if args.delete:
        execute_deletions(delete_plan, args.log_dir_root, args.dry)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
