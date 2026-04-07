#!/usr/bin/env bash

set -euo pipefail

PY_SCRIPT="scripts/data/run_hest1k.py"

[[ -f "${PY_SCRIPT}" ]] || {
    echo "Missing processing script: ${PY_SCRIPT}" >&2
    exit 1
}

cmd=(python "${PY_SCRIPT}" --executor=ray "$@")
printf -v remote_cmd '%q ' "${cmd[@]}"
remote_cmd="${remote_cmd% }"

"ray/submit.sh" "${remote_cmd}"
