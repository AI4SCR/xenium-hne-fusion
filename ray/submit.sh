#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_TEMPLATE="${REPO_ROOT}/ray/runtime_envs/runtime_env_template.yml"
ENV_FILE="${REPO_ROOT}/ray/runtime_envs/runtime_env.yml"
XENIUM_MODULE_PATH="$(cd "${REPO_ROOT}/src/xenium_hne_fusion" && pwd -P)"
AI4BMR_MODULE_PATH="$(cd "${REPO_ROOT}/ray/other_modules/ai4bmr-learn/src/ai4bmr_learn" && pwd -P)"

if [[ -f "${REPO_ROOT}/.env.kaiko" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env.kaiko"
    set +a
fi

[[ -n "${WANDB_API_KEY:-}" ]] || {
    echo "WANDB_API_KEY missing in .env.kaiko" >&2
    exit 1
}
[[ -f "${ENV_TEMPLATE}" ]] || {
    echo "Missing runtime env template: ${ENV_TEMPLATE}" >&2
    exit 1
}
[[ -d "${XENIUM_MODULE_PATH}" ]] || {
    echo "Missing module path: ${XENIUM_MODULE_PATH}" >&2
    exit 1
}
[[ -d "${AI4BMR_MODULE_PATH}" ]] || {
    echo "Missing module path: ${AI4BMR_MODULE_PATH}" >&2
    exit 1
}
command -v envsubst >/dev/null 2>&1 || {
    echo "envsubst not found on PATH." >&2
    exit 1
}

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export KAIKO_XENIUM_MODULE_PATH="${XENIUM_MODULE_PATH}"
export KAIKO_AI4BMR_MODULE_PATH="${AI4BMR_MODULE_PATH}"
envsubst < "${ENV_TEMPLATE}" > "${ENV_FILE}"

RAY_ADDRESS="https://chuv.nebul.prd.kaiko.ai"
ENTRYPOINT_NUM_GPUS="${KAIKO_ENTRYPOINT_NUM_GPUS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --entrypoint-num-gpus)
            shift
            [[ $# -gt 0 ]] || {
                echo "Missing value for --entrypoint-num-gpus" >&2
                exit 1
            }
            ENTRYPOINT_NUM_GPUS="$1"
            shift
            ;;
        --entrypoint-num-gpus=*)
            ENTRYPOINT_NUM_GPUS="${1#*=}"
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: ray/submit.sh [--entrypoint-num-gpus N] [--] [REMOTE_CMD...]

Defaults to --entrypoint-num-gpus 0 and runs `pwd` when no remote command is given.
EOF
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

REMOTE_CMD="${*:-pwd}"

kray job submit \
    --address "$RAY_ADDRESS" \
    --entrypoint-num-gpus "$ENTRYPOINT_NUM_GPUS" \
    --working-dir "${REPO_ROOT}" \
    --runtime-env "${ENV_FILE}" \
    -- bash -c "$REMOTE_CMD"
