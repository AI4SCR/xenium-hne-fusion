# MIL Slurm Commands

## Overview

MIL (Multiple Instance Learning) training runs on top of a pretrained supervised model.
The pipeline has three stages:

1. **Clinical artifact** — join `cells.json` items with structured sample-level metadata
   to produce `artifacts/mil/cells/clinical.parquet`. This file maps each `sample_id` to
   its clinical labels and is used as the target source during MIL training.

2. **Prediction cache** — the pretrained model runs inference over all tiles (all splits)
   and caches per-tile embeddings. Embeddings are regrouped by `sample_id` and written to
   `bags.json` — one entry per patient bag with a `path` pointing to a stacked `.pt` file.
   This step must be run before training and is submitted as a GPU job.

3. **MIL training** — `MILBagsDataset` loads `bags.json`, filters bags by split, joins
   clinical labels from `clinical.parquet`, and trains an attention aggregator + head with
   PyTorch Lightning. Bags are collated with padding (`pad_bags_collate`) to handle
   variable bag sizes. Training asserts `bags.json` exists and fails early if step 2 was skipped.

## Data Preparation

```bash
# Step 1 — clinical artifact (run once per dataset / items variant)
uv run python scripts/artifacts/create_mil_metadata.py --config configs/mil/beat/classification.yaml

# Step 2 — prediction cache (run once per pretrained run)
PARTITION=gpu-l40
ACCOUNT=mrapsoma_prometex
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

Step 2 writes `bags.json` and `<sample_id>.pt` files to `DATA_DIR/03_output/beat/mil/<run_name>/predictions/`.

## Model Training

```bash
PARTITION=gpu-l40
ACCOUNT=mrapsoma_prometex
CONFIG=configs/mil/beat/classification.yaml

for RUN_ID in \
    2d8h8hr7 9ej3jw59 dvc1qwis lrqyfqta \
    0bt8b9ov hntkhq7e f2nbmz9r y2m7ge73 \
    6vutoy7v waqirmiu 8lu2tzqi xp3znjz0 \
    saib5muk uq3oth9w w9v9kgtn 59xu56ws; do
    for AGGREGATOR in mean max min simple_attention attention; do
        sbatch \
            --account=${ACCOUNT} \
            --cpus-per-task=12 \
            --mem=64G \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=04:30:00 \
            --output=$HOME/logs/%j.out \
            --job-name=mil-${RUN_ID}-${AGGREGATOR} \
            --wrap="uv run python scripts/train/mil.py \
                --config ${CONFIG} \
                --pretrained.run_id ${RUN_ID} \
                --aggregator.name ${AGGREGATOR}"
    done
done
```

Training will fail immediately if `bags.json` does not exist (step 2 not yet run).

## Config Layout

```
configs/mil/<dataset>/<task>.yaml
```

Key fields in the config:

| Field | Description |
|-------|-------------|
| `pretrained.entity/project/run_id` | W&B run of the pretrained supervised model |
| `data.metadata_path` | Tile-level split parquet (relative to `03_output/<name>/splits/`) |
| `data.clinical_path` | Sample-level clinical labels (relative to `03_output/<name>/`) |
| `data.num_workers` | Workers used for both prediction and MIL training dataloaders |
| `lit.target_key` | Column to predict, prefixed with `metadata.` (e.g. `metadata.7`) |
| `task.kind` | `classification` or `regression` |


### SLURM submission

```bash
PARTITION=gpu-l40
ACCOUNT=mrapsoma_prometex
CONFIG=configs/mil/beat/classification.yaml

for RUN_ID in \
    2d8h8hr7 9ej3jw59 dvc1qwis lrqyfqta \
    0bt8b9ov hntkhq7e f2nbmz9r y2m7ge73 \
    6vutoy7v waqirmiu 8lu2tzqi xp3znjz0 \
    saib5muk uq3oth9w w9v9kgtn 59xu56ws; do
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
            --config configs/mil/beat/${RUN_ID}.yaml")

    for AGGREGATOR in mean max min simple_attention attention; do
        sbatch \
            --dependency=afterok:${CACHE_JID} \
            --account=${ACCOUNT} \
            --cpus-per-task=12 \
            --mem=128G \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=04:30:00 \
            --output=$HOME/logs/%j.out \
            --job-name=mil-${RUN_ID}-${AGGREGATOR} \
            --wrap="uv run python scripts/train/mil.py \
                --config ${CONFIG} \
                --pretrained.run_id ${RUN_ID} \
                --aggregator.name ${AGGREGATOR}"
    done
done
```

Each cache job writes `bags.json` and `<sample_id>.pt` files to
`DATA_DIR/03_output/beat/mil/<run_id>/predictions/`.
Training jobs are held in `(Dependency)` state until the cache job completes successfully.

