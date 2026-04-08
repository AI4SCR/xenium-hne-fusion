#!/usr/bin/env bash

SLURM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_REPO_ROOT="$(cd "${SLURM_SCRIPT_DIR}/.." && pwd)"
SLURM_LOG_DIR="${HOME}/logs"


slurm_load_env() {
    if [[ -f "${SLURM_REPO_ROOT}/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "${SLURM_REPO_ROOT}/.env"
        set +a
    fi
    mkdir -p "${SLURM_LOG_DIR}"
}


slurm_require_file() {
    local path="$1"
    [[ -f "${path}" ]] || {
        echo "Missing file: ${path}" >&2
        exit 1
    }
}


slurm_require_dir() {
    local path="$1"
    [[ -d "${path}" ]] || {
        echo "Missing directory: ${path}" >&2
        exit 1
    }
}


slurm_require_env_dir() {
    local var_name="$1"
    local path="${!var_name:-}"
    [[ -n "${path}" ]] || {
        echo "${var_name} is not set." >&2
        exit 1
    }
    slurm_require_dir "${path}"
}


slurm_wrap_cmd() {
    local wrapped_cmd
    printf -v wrapped_cmd '%q ' "$@"
    printf '%s\n' "${wrapped_cmd% }"
}


slurm_submit_job() {
    local job_name="$1"
    local dependency="$2"
    local gres="$3"
    shift 3

    local wrapped_cmd
    wrapped_cmd="$(slurm_wrap_cmd "$@")"

    local sbatch_args=(
        --parsable
        --job-name "${job_name}"
        --cpus-per-task 8
        --mem 64G
        --time 08:00:00
        --output "${SLURM_LOG_DIR}/%j.log"
        --error "${SLURM_LOG_DIR}/%j.err"
    )
    if [[ -n "${dependency}" ]]; then
        sbatch_args+=(--dependency "${dependency}")
    fi
    if [[ -n "${gres}" ]]; then
        sbatch_args+=(--gres "${gres}")
    fi

    sbatch "${sbatch_args[@]}" --wrap "${wrapped_cmd}"
}


slurm_submit_cpu_job() {
    local job_name="$1"
    local dependency="$2"
    shift 2
    slurm_submit_job "${job_name}" "${dependency}" "" "$@"
}


slurm_submit_gpu_job() {
    local job_name="$1"
    local dependency="$2"
    shift 2
    slurm_submit_job "${job_name}" "${dependency}" "gpu:1" "$@"
}


slurm_afterok_dependency() {
    local IFS=:
    printf 'afterok:%s\n' "$*"
}
