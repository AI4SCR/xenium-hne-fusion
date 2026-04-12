#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../slurm/common.sh
source "${REPO_ROOT}/slurm/common.sh"

CONFIG_PATH=""
CPUS=8
MEM="64G"
TIME="08:00:00"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  scribble/sbatch_process_beat_cells.sh --config configs/data/remote/beat.yaml [--dry-run]

Options:
  --config PATH   BEAT data config.
  --cpus N        CPUs per sample job. Default: 8
  --mem MEM       Memory per sample job. Default: 64G
  --time TIME     Walltime per sample job. Default: 08:00:00
  --dry-run       Print sbatch commands without submitting.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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
slurm_require_env_dir BEAT_RAW_DIR
cd "${REPO_ROOT}"

SAMPLE_IDS=()
while IFS= read -r sample_id; do
    SAMPLE_IDS+=("${sample_id}")
done < <(find "${BEAT_RAW_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No BEAT sample directories found in ${BEAT_RAW_DIR}" >&2
    exit 1
}

for sample_id in "${SAMPLE_IDS[@]}"; do
    cmd=(
        uv run python scribble/sbatch_process_beat_cells.py
        --config "${CONFIG_PATH}"
        --sample-id "${sample_id}"
        --worker
    )

    if [[ "${DRY_RUN}" == true ]]; then
        wrapped_cmd="$(slurm_wrap_cmd "${cmd[@]}")"
        echo sbatch --parsable --job-name "beat_cells_${sample_id}" --cpus-per-task "${CPUS}" --mem "${MEM}" --time "${TIME}" --output "${SLURM_LOG_DIR}/%j.log" --error "${SLURM_LOG_DIR}/%j.err" --wrap "${wrapped_cmd}"
        continue
    fi

    job_id="$(
        sbatch \
            --parsable \
            --job-name "beat_cells_${sample_id}" \
            --cpus-per-task "${CPUS}" \
            --mem "${MEM}" \
            --time "${TIME}" \
            --output "${SLURM_LOG_DIR}/%j.log" \
            --error "${SLURM_LOG_DIR}/%j.err" \
            --wrap "$(slurm_wrap_cmd "${cmd[@]}")"
    )"
    job_id="${job_id%%;*}"
    echo "Submitted ${sample_id}: ${job_id}"
done
