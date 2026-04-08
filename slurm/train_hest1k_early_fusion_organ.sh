#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm/common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./slurm/train_hest1k_early_fusion_organ.sh <breast|lung|pancreas>
EOF
}

[[ $# -eq 1 ]] || {
    usage >&2
    exit 1
}

ORGAN="$1"
case "${ORGAN}" in
    breast|lung|pancreas)
        ;;
    *)
        echo "Unknown organ: ${ORGAN}" >&2
        usage >&2
        exit 1
        ;;
esac

CONFIG_PATH="$(realpath "configs/train/hest1k/expression/${ORGAN}/early-fusion.yaml")"
slurm_load_env
slurm_require_file "${CONFIG_PATH}"

cmd=(
    uv run python
    scripts/train/supervised.py
    --config "${CONFIG_PATH}"
)

job_id="$(slurm_submit_gpu_job "hest1k_early_fusion_${ORGAN}" "" "${cmd[@]}")"
job_id="${job_id%%;*}"
echo "Submitted ${ORGAN} early-fusion training: ${job_id}"
