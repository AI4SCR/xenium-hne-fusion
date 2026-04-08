#!/usr/bin/env bash

set -euo pipefail

PY_SCRIPT="scripts/data/run_beat.py"
CONFIG_PATH=""

usage() {
    cat <<'EOF'
Usage:
  ./ray/scripts/run_beat.sh --config configs/data/remote/beat.yaml
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

[[ -f "${PY_SCRIPT}" ]] || {
    echo "Missing processing script: ${PY_SCRIPT}" >&2
    exit 1
}

[[ -f "${CONFIG_PATH}" ]] || {
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
}

cmd=(python "${PY_SCRIPT}" --executor=ray --config "${CONFIG_PATH}")
printf -v remote_cmd '%q ' "${cmd[@]}"
remote_cmd="${remote_cmd% }"

"ray/submit.sh" "${remote_cmd}"
