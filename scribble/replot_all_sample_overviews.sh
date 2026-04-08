#!/usr/bin/env bash

set -euo pipefail

N="${N:-10000}"
MAX_SIZE="${MAX_SIZE:-2048}"
SEED="${SEED:-0}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${DATA_DIR:?DATA_DIR must be set in .env or the environment}"
ROOT="${DATA_DIR}/01_structured"

for dataset in beat hest1k; do
  dataset_dir="${ROOT}/${dataset}"
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Skipping missing dataset dir: ${dataset_dir}"
    continue
  fi

  for sample_dir in "${dataset_dir}"/*; do
    if [[ ! -d "${sample_dir}" ]]; then
      continue
    fi

    echo "Replotting ${sample_dir}"
    uv run python scribble/replot_sample_overview.py \
      "${sample_dir}" \
      --n "${N}" \
      --max_size "${MAX_SIZE}" \
      --seed "${SEED}"
  done
done
