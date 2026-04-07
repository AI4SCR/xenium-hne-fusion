#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/configs/data/remote/hest1k.yaml"

usage() {
    cat <<'EOF'
Usage:
  scripts/data/run_hest1k_slurm.sh [--config configs/data/remote/hest1k.yaml]
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

mapfile -t SAMPLE_IDS < <(
    uv run python -c '
from pathlib import Path
import os
import sys
from xenium_hne_fusion.utils.getters import load_processing_config, resolve_samples

config_path = Path(sys.argv[1])
metadata_path = Path(os.environ["HEST1K_RAW_DIR"]) / "HEST_v1_3_0.csv"
cfg = load_processing_config(config_path)
for sample_id in resolve_samples(cfg, metadata_path):
    print(sample_id)
' "${CONFIG_PATH}"
)

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No HEST1K sample IDs resolved from ${CONFIG_PATH}" >&2
    exit 1
}

for sample_id in "${SAMPLE_IDS[@]}"; do
    cmd=(
        uv run
        scripts/data/run_hest1k.py
        --config "${CONFIG_PATH}"
        --name hest1k
        --executor serial
        --filter.sample_ids "[${sample_id}]"
    )

    printf -v wrapped_cmd '%q ' "${cmd[@]}"
    wrapped_cmd="${wrapped_cmd% }"

    echo "Submitting ${sample_id}"
    sbatch \
        --job-name "hest1k_${sample_id}" \
        --cpus-per-task 8 \
        --mem 64G \
        --time 08:00:00 \
        --output "${LOG_DIR}/%j.log" \
        --error "${LOG_DIR}/%j.err" \
        --wrap "${wrapped_cmd}"
done
