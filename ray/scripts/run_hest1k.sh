#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="scripts/data/run_hest1k.py"

[[ -f "${REPO_ROOT}/${PY_SCRIPT}" ]] || {
    echo "Missing processing script: ${REPO_ROOT}/${PY_SCRIPT}" >&2
    exit 1
}

cmd=(python "${PY_SCRIPT}" "$@")
printf -v remote_cmd '%q ' "${cmd[@]}"
remote_cmd="${remote_cmd% }"

"${REPO_ROOT}/ray/submit.sh" "${remote_cmd}"
