#!/usr/bin/env bash

set -euo pipefail

KEEP_1="BB3833A4-DBE7-473E-A32E-88B06D222A28"
KEEP_2="6B356E0F-1926-434F-9E02-C849020C9E74"
ADDR="https://chuv.nebul.prd.kaiko.ai"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "[DRY-RUN] No jobs will be stopped"
fi

get_running_jobs() {
  kray job list --address "$ADDR" | awk '
    /^- user:/ {
      sub_id=""
      status=""
    }
    /^[[:space:]]*submission_id:/ {
      sub_id=$2
    }
    /^[[:space:]]*status:/ {
      status=$2
      if (status == "RUNNING" && sub_id != "") {
        print sub_id
      }
    }
  ' | sort -u
}

prev="__INIT__"

while true; do
  current="$(get_running_jobs)"

  while IFS= read -r job_id; do
    [[ -z "$job_id" ]] && continue

    case "$job_id" in
      "$KEEP_1"|"$KEEP_2")
        ;;
      *)
        if [[ $DRY_RUN -eq 1 ]]; then
          echo "[DRY-RUN] Would stop submission_id=$job_id"
        else
          echo "Stopping submission_id=$job_id"
          kray job stop "$job_id" --address "$ADDR"
        fi
        ;;
    esac
  done <<< "$current"

  sleep 2

  new_current="$(get_running_jobs)"

  if [[ "$new_current" == "$prev" ]]; then
    echo "No new jobs starting. Done."
    break
  fi

  prev="$new_current"
done