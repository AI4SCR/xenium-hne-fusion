#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="${HOME}/logs"

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

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

mkdir -p "${LOG_DIR}"

CONFIG_PATH="$(realpath "configs/train/hest1k/expression/${ORGAN}/early-fusion.yaml")"

[[ -f "${CONFIG_PATH}" ]] || {
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
}

cmd=(
    uv run
    scripts/train/supervised.py
    --config "${CONFIG_PATH}"
)

printf -v wrapped_cmd '%q ' "${cmd[@]}"
wrapped_cmd="${wrapped_cmd% }"

echo "Submitting ${ORGAN} early-fusion training"
sbatch \
    --job-name "hest1k_early_fusion_${ORGAN}" \
    --cpus-per-task 8 \
    --mem 64G \
    --time 08:00:00 \
    --output "${LOG_DIR}/%j.log" \
    --error "${LOG_DIR}/%j.err" \
    --wrap "${wrapped_cmd}"
