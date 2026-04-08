#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm/common.sh
source "${SCRIPT_DIR}/common.sh"

CONFIG_PATH=""
OVERWRITE=false

usage() {
    cat <<'EOF'
Usage:
  ./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml [--overwrite]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -n "${CONFIG_PATH}" ]] || {
    echo "--config is required." >&2
    usage >&2
    exit 1
}

CONFIG_PATH="$(realpath "${CONFIG_PATH}")"

slurm_load_env
slurm_require_file "${CONFIG_PATH}"
slurm_require_env_dir HEST1K_RAW_DIR
slurm_require_file "${HEST1K_RAW_DIR}/HEST_v1_3_0.csv"

SAMPLE_IDS=()
while IFS= read -r sample_id; do
    SAMPLE_IDS+=("${sample_id}")
done < <(uv run python scripts/data/list_samples.py --config "${CONFIG_PATH}")

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No HEST1K samples matched the config." >&2
    exit 1
}

JOB_IDS=()
for sample_id in "${SAMPLE_IDS[@]}"; do
    cmd=(
        uv run python scripts/data/run_hest1k.py
        --config "${CONFIG_PATH}"
        --executor serial
        --stage samples
        --filter.include_ids "[${sample_id}]"
        --filter.exclude_ids null
    )
    if [[ "${OVERWRITE}" == true ]]; then
        cmd+=(--overwrite true)
    fi

    job_id="$(slurm_submit_cpu_job "hest1k_${sample_id}" "" "${cmd[@]}")"
    job_id="${job_id%%;*}"
    JOB_IDS+=("${job_id}")
    echo "Submitted ${sample_id}: ${job_id}"
done

dependency="$(slurm_afterok_dependency "${JOB_IDS[@]}")"
final_cmd=(
    uv run python scripts/data/run_hest1k.py
    --config "${CONFIG_PATH}"
    --executor serial
    --stage finalize
)
if [[ "${OVERWRITE}" == true ]]; then
    final_cmd+=(--overwrite true)
fi

final_job_id="$(slurm_submit_cpu_job "hest1k_finalize" "${dependency}" "${final_cmd[@]}")"
final_job_id="${final_job_id%%;*}"
echo "Submitted finalization: ${final_job_id}"
