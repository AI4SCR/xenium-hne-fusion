#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH=""

usage() {
    cat <<'EOF'
Usage:
  ./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml
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

[[ -n "${CONFIG_PATH}" ]] || {
    echo "--config is required." >&2
    usage >&2
    exit 1
}

CONFIG_PATH="$(realpath "${CONFIG_PATH}")"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"

[[ -f "${CONFIG_PATH}" ]] || {
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
}

[[ -n "${HEST1K_RAW_DIR:-}" ]] || {
    echo "HEST1K_RAW_DIR is not set." >&2
    exit 1
}

[[ -d "${HEST1K_RAW_DIR}" ]] || {
    echo "HEST1K_RAW_DIR does not exist: ${HEST1K_RAW_DIR}" >&2
    exit 1
}

[[ -f "${HEST1K_RAW_DIR}/HEST_v1_3_0.csv" ]] || {
    echo "Missing HEST metadata: ${HEST1K_RAW_DIR}/HEST_v1_3_0.csv" >&2
    exit 1
}

cmd=(
    uv run python
    scripts/data/process_hest1k.py
    --config "${CONFIG_PATH}"
)

printf -v wrapped_cmd '%q ' "${cmd[@]}"
wrapped_cmd="${wrapped_cmd% }"

echo "Submitting HEST1K processing job"
sbatch \
    --job-name "hest1k_process" \
    --cpus-per-task 8 \
    --mem 64G \
    --time 08:00:00 \
    --output "${LOG_DIR}/%j.log" \
    --error "${LOG_DIR}/%j.err" \
    --wrap "${wrapped_cmd}"
