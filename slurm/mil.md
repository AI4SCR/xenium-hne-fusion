# MIL Slurm Commands

## Overview

MIL (Multiple Instance Learning) training runs on top of a pretrained supervised model.
The pipeline has two stages:

1. **Prediction cache** — the pretrained model runs inference over all tiles (all splits)
   and caches per-tile embeddings. Embeddings are regrouped by `sample_id` and written to
   `bags.json` — one entry per patient bag with a `path` pointing to a stacked `.pt` file.
   This step must be run before training and is submitted as a GPU job.

2. **MIL training** — on first run, `train()` automatically builds `metadata.parquet` in the
   run directory by joining the pretrained run's supervised split parquet with structured
   sample-level metadata from `01_structured/metadata.parquet`. This file carries `split` and
   all clinical columns. `MILBagsDataset` loads `bags.json`, filters bags by split, joins
   clinical labels, and trains an attention aggregator + head with PyTorch Lightning. Bags are
   collated with padding (`pad_bags_collate`) to handle variable bag sizes. Training asserts
   `bags.json` exists and fails early if step 1 was skipped.

## Data Preparation

```bash
# Step 1 — prediction cache (run once per pretrained run)
PARTITION=gpu-h100  # previously gpu-l40
ACCOUNT=rgottar1_spatial  # previously mrapsoma_prometex (poor fairshare)
sbatch \
    --account=${ACCOUNT} \
    --cpus-per-task=12 \
    --mem=128G \
    --gres=gpu:1 \
    --partition=${PARTITION} \
    --time=06:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=mil-cache-beat \
    --wrap="uv run python scripts/artifacts/cache_predictions.py \
        --config configs/mil/beat/classification.yaml"
```

Step 1 writes `bags.json` and `<sample_id>.pt` files to `DATA_DIR/03_output/beat/mil/<run_id>/predictions/`.

## SLURM Submission

Run cache + training for all pretrained runs. Cache and training jobs are chained with
`afterok` dependency — training starts automatically once the cache is ready.

```bash
PARTITION=gpu-h100  # previously gpu-l40
ACCOUNT=rgottar1_spatial  # previously mrapsoma_prometex (poor fairshare)
CONFIG=configs/mil/beat/classification.yaml

# xe-hne-fus-cell-v1 finished runs — one per (model, outer fold, freeze) combination
# freeze=False
RUN_IDS_UNFREEZE=(
    40utmvmw j5x4suw2 owzcohia fif5wuld   # early-fusion       outer 0-3
    lzj2k6yd n69cixtr bp2la5tg oblzes57   # expr-tile          outer 0-3
    m886lfhu 3pqmw7gh yeqi6amk jr7uo9ng   # expr-token         outer 0-3
    w7olnchy lbwctiyq p4feamts 5zwl441m   # late-fusion-tile   outer 0-3
    w91hb1aw as1wwjmf vur2wcn3 uglyuryw   # late-fusion-token  outer 0-3
    icxsfpqf ua31ay2l cdr1rg1u gnre6bzb   # vision             outer 0-3
)

# freeze=True (multimodal models only)
RUN_IDS_FREEZE=(
    6irt7oje lb031j0q rw1fei6z a4oj0ra0   # early-fusion       outer 0-3
    y5sz738k ga578srh 1gjis6ut vjxjk7uv   # late-fusion-tile   outer 0-3
    4a5wzxi5 6p2905h9 4cl2unb7 ck706f0j   # late-fusion-token  outer 0-3
    bwqxlmkm oa90ekqc 0glswn1l t27q3v41   # vision             outer 0-3
)

RUN_IDS=("${RUN_IDS_UNFREEZE[@]}" "${RUN_IDS_FREEZE[@]}")

for RUN_ID in "${RUN_IDS[@]}"; do
    CACHE_JID=$(sbatch --parsable \
        --account=${ACCOUNT} \
        --cpus-per-task=12 \
        --mem=128G \
        --gres=gpu:1 \
        --partition=${PARTITION} \
        --time=06:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=mil-cache-${RUN_ID} \
        --wrap="uv run python scripts/artifacts/cache_predictions.py \
            --config ${CONFIG} \
            --pretrained.run_id ${RUN_ID}")

    for AGGREGATOR in mean max min simple_attention attention; do
        sbatch \
            --dependency=afterok:${CACHE_JID} \
            --account=${ACCOUNT} \
            --cpus-per-task=12 \
            --mem=64G \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=02:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=mil-${RUN_ID}-${AGGREGATOR} \
            --wrap="uv run python scripts/train/mil.py \
                --config ${CONFIG} \
                --pretrained.run_id ${RUN_ID} \
                --aggregator.name ${AGGREGATOR}"
    done
done
```

Each cache job writes `bags.json` and `<sample_id>.pt` to
`DATA_DIR/03_output/beat/mil/<run_id>/predictions/`.
Training jobs are held in `(Dependency)` state until the cache job completes successfully.

## Config Layout

```
configs/mil/<dataset>/<task>.yaml
```

Key fields in the config:

| Field | Description |
|-------|-------------|
| `pretrained.entity/project/run_id` | W&B run of the pretrained supervised model |
| `data.metadata_path` | Optional absolute path to custom sample-level metadata (with `split` column). If `null`, built automatically from supervised split parquet + `01_structured/metadata.parquet`. |
| `data.num_workers` | Workers used for both prediction and MIL training dataloaders |
| `lit.target_key` | Column to predict, prefixed with `metadata.` (e.g. `metadata.7`) |
| `task.kind` | `classification` or `regression` |
| `task.num_classes` | Number of classes for classification tasks (required when `task.kind=classification`) |