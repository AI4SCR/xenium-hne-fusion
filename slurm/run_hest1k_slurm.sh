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

mapfile -t SAMPLE_IDS < <(
    uv run python -c '
from pathlib import Path
import sys
from xenium_hne_fusion.utils.getters import load_processing_config, resolve_samples

cfg = load_processing_config(Path(sys.argv[1]))
print("\n".join(resolve_samples(cfg, Path(sys.argv[2]))))
' "${CONFIG_PATH}" "${HEST1K_RAW_DIR}/HEST_v1_3_0.csv"
)

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No HEST1K samples matched the config." >&2
    exit 1
}

for sample_id in "${SAMPLE_IDS[@]}"; do
    cmd=(
        bash -lc
        "set -euo pipefail; \
tmp_config=\$(mktemp \"${LOG_DIR}/hest1k_${sample_id}.XXXX.yaml\"); \
trap 'rm -f \"${tmp_config}\"' EXIT; \
uv run python -c 'from pathlib import Path; import sys, yaml; src = Path(sys.argv[1]); sample_id = sys.argv[2]; dst = Path(sys.argv[3]); data = yaml.safe_load(src.read_text()) or {}; filter_cfg = data.setdefault(\"filter\", {}); filter_cfg[\"include_ids\"] = [sample_id]; filter_cfg[\"exclude_ids\"] = None; dst.write_text(yaml.safe_dump(data, sort_keys=False))' \"${CONFIG_PATH}\" \"${sample_id}\" \"${tmp_config}\"; \
uv run python scripts/data/process_hest1k.py --config \"${tmp_config}\""
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
