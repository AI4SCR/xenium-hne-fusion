#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./ray/scripts/run_data.sh --config CONFIG_PATH [SCRIPT_ARGS...]

Examples:
  ./ray/scripts/run_data.sh --config configs/data/remote/beat.yaml
  ./ray/scripts/run_data.sh --config configs/data/remote/hest1k.yaml --overwrite true

The dataset script is derived from `name:` in the config file.
All arguments are forwarded unchanged to the underlying
Python script with `--executor ray` added automatically.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi

ARGS=("$@")
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            shift
            [[ $# -gt 0 ]] || {
                echo "Missing value for --config." >&2
                exit 1
            }
            CONFIG_PATH="$1"
            ;;
        --config=*)
            CONFIG_PATH="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
    esac
    shift
done

[[ -n "${CONFIG_PATH}" ]] || {
    echo "--config is required." >&2
    usage >&2
    exit 1
}

[[ -f "${CONFIG_PATH}" ]] || {
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
}

DATASET="$(awk -F: '/^name:[[:space:]]*/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "${CONFIG_PATH}")"

case "${DATASET}" in
    beat|hest1k)
        ;;
    "")
        echo "Missing top-level name in config: ${CONFIG_PATH}" >&2
        exit 1
        ;;
    *)
        echo "Unsupported dataset in ${CONFIG_PATH}: ${DATASET}" >&2
        exit 1
        ;;
esac

PY_SCRIPT="scripts/data/run_${DATASET}.py"
[[ -f "${PY_SCRIPT}" ]] || {
    echo "Missing processing script: ${PY_SCRIPT}" >&2
    exit 1
}

exec "ray/submit.sh" python "${PY_SCRIPT}" --executor ray "${ARGS[@]}"
