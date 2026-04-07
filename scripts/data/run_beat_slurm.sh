#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/configs/data/remote/beat.yaml"

usage() {
    cat <<'EOF'
Usage:
  scripts/data/run_beat_slurm.sh [--config configs/data/remote/beat.yaml]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
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

cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

[[ -f "${CONFIG_PATH}" ]] || {
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
}

[[ -n "${BEAT_RAW_DIR:-}" ]] || {
    echo "BEAT_RAW_DIR is not set." >&2
    exit 1
}

[[ -d "${BEAT_RAW_DIR}" ]] || {
    echo "BEAT_RAW_DIR does not exist: ${BEAT_RAW_DIR}" >&2
    exit 1
}

mapfile -t SAMPLE_IDS < <(
    find "${BEAT_RAW_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
)

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No BEAT sample directories found in ${BEAT_RAW_DIR}" >&2
    exit 1
}

for sample_id in "${SAMPLE_IDS[@]}"; do
    cmd=(
        uv run
        scripts/data/run_beat.py
        --config "${CONFIG_PATH}"
        --name beat
        --executor serial
        --filter.sample_ids "[${sample_id}]"
    )

    printf -v wrapped_cmd '%q ' "${cmd[@]}"
    wrapped_cmd="${wrapped_cmd% }"

    echo "Submitting ${sample_id}"
    sbatch \
        --job-name "beat_${sample_id}" \
        --cpus-per-task 8 \
        --time 08:00:00 \
        --wrap "${wrapped_cmd}"
done
