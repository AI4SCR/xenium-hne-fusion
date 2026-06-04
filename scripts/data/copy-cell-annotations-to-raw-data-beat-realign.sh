#!/usr/bin/env bash
set -euo pipefail

# Enable recursive globbing
shopt -s globstar

if [ -f "./.env" ]; then
    # Read lines, ignore comments, and export them
    export $(grep -v '^#' ./.env | xargs)
else
    echo "Error: .env file not found in the current directory ($(pwd))" >&2
    exit 1
fi

echo "Looking for cell annotations at ${BEAT_CELL_ANNOTATIONS_DIR}"
# 1. Glob all the files into an array
files=("${BEAT_CELL_ANNOTATIONS_DIR}"/**/centroid_cropped.parquet)

# Handle case where no files match (globstar leaves the string intact if unmatched)
if [[ ! -e "${files[0]}" ]]; then
    files=()
fi

# 2. Print the number of found files
file_count=${#files[@]}
echo "Found ${file_count} 'centroid_cropped.parquet' files."

# 3. Proceed with the loop
for src_path in "${files[@]}"; do
    sample_id="$(basename "$(dirname "$(dirname "${src_path}")")")"
    dst_dir="${BEAT_RAW_DIR}/${sample_id}"

    if [[ ! -d "${dst_dir}" ]]; then
        echo "Missing sample dir: ${dst_dir}" >&2
        exit 1
    fi

    cp "${src_path}" "${dst_dir}/cells_.parquet"
    echo "Copied ${src_path} -> ${dst_dir}/cells_.parquet"
done