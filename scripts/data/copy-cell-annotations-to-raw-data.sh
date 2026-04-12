#!/usr/bin/env bash
set -euo pipefail

for src_path in "${BEAT_CELL_ANNOTATIONS_DIR}"/*.parquet; do
    sample_id="$(basename "${src_path}" .parquet)"
    dst_dir="${BEAT_RAW_DIR}/${sample_id}"

    if [[ ! -d "${dst_dir}" ]]; then
        echo "Missing sample dir: ${dst_dir}" >&2
        exit 1
    fi

    cp "${src_path}" "${dst_dir}/cells.parquet"
    echo "Copied ${src_path} -> ${dst_dir}/cells.parquet"
done
