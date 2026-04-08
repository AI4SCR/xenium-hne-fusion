#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH=""

usage() {
    cat <<'EOF'
Usage:
  ./slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
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

[[ -n "${BEAT_RAW_DIR:-}" ]] || {
    echo "BEAT_RAW_DIR is not set." >&2
    exit 1
}

[[ -d "${BEAT_RAW_DIR}" ]] || {
    echo "BEAT_RAW_DIR does not exist: ${BEAT_RAW_DIR}" >&2
    exit 1
}

[[ -n "${DATA_DIR:-}" ]] || {
    echo "DATA_DIR is not set." >&2
    exit 1
}

SAMPLE_IDS=()
while IFS= read -r sample_id; do
    SAMPLE_IDS+=("${sample_id}")
done < <(
    find "${BEAT_RAW_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || {
    echo "No BEAT sample directories found in ${BEAT_RAW_DIR}" >&2
    exit 1
}

read -r DATASET_NAME TILE_PX STRIDE_PX TILE_MPP KERNEL_SIZE PREDICATE <<<"$(
    uv run python -c '
from pathlib import Path
import sys
from xenium_hne_fusion.utils.getters import load_pipeline_config

cfg = load_pipeline_config("beat", Path(sys.argv[1]))
tiles = cfg.processing.tiles
print(cfg.name, tiles.tile_px, tiles.stride_px, tiles.mpp, tiles.kernel_size, tiles.predicate)
' "${CONFIG_PATH}"
)"

for sample_id in "${SAMPLE_IDS[@]}"; do
    structured_dir="${DATA_DIR}/01_structured/${DATASET_NAME}/${sample_id}"
    processed_dir="${DATA_DIR}/02_processed/${DATASET_NAME}/${sample_id}/${TILE_PX}_${STRIDE_PX}"
    wsi_path="${structured_dir}/wsi.tiff"
    transcripts_path="${structured_dir}/transcripts.parquet"
    cells_path="${structured_dir}/cells.parquet"
    tissues_path="${structured_dir}/tissues.parquet"
    tiles_path="${structured_dir}/tiles/${TILE_PX}_${STRIDE_PX}.parquet"

    cmd=(
        bash -lc
        "set -euo pipefail; \
if [[ ! -f \"${wsi_path}\" ]]; then echo \"Missing WSI: ${wsi_path}\" >&2; exit 1; fi; \
if [[ ! -f \"${transcripts_path}\" ]]; then echo \"Missing transcripts: ${transcripts_path}\" >&2; exit 1; fi; \
uv run python scripts/data/detect_tissues.py --wsi_path \"${wsi_path}\" --output_parquet \"${tissues_path}\"; \
uv run python scripts/data/tile.py --wsi_path \"${wsi_path}\" --tissues_parquet \"${tissues_path}\" --output_parquet \"${tiles_path}\" --tile_px ${TILE_PX} --stride_px ${STRIDE_PX} --mpp ${TILE_MPP}; \
if [[ -f \"${cells_path}\" ]]; then \
  uv run python scripts/data/process.py --wsi_path \"${wsi_path}\" --tiles_parquet \"${tiles_path}\" --transcripts_path \"${transcripts_path}\" --output_dir \"${processed_dir}\" --mpp ${TILE_MPP} --predicate \"${PREDICATE}\" --img_size ${TILE_PX} --kernel_size ${KERNEL_SIZE} --cells_path \"${cells_path}\"; \
else \
  uv run python scripts/data/process.py --wsi_path \"${wsi_path}\" --tiles_parquet \"${tiles_path}\" --transcripts_path \"${transcripts_path}\" --output_dir \"${processed_dir}\" --mpp ${TILE_MPP} --predicate \"${PREDICATE}\" --img_size ${TILE_PX} --kernel_size ${KERNEL_SIZE}; \
fi"
    )

    printf -v wrapped_cmd '%q ' "${cmd[@]}"
    wrapped_cmd="${wrapped_cmd% }"

    echo "Submitting ${sample_id}"
    sbatch \
        --job-name "beat_${sample_id}" \
        --cpus-per-task 8 \
        --mem 64G \
        --time 08:00:00 \
        --output "${LOG_DIR}/%j.log" \
        --error "${LOG_DIR}/%j.err" \
        --wrap "${wrapped_cmd}"
done
