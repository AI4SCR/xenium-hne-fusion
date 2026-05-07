# BEAT Slurm Commands


## Data Preparation

Steps must run in order; each stage depends on the previous one.

```bash
# 1. Transfer cell annotations into the raw data directory.
chmod u+x scripts/data/copy-cell-annotations-to-raw-data.sh && scripts/data/copy-cell-annotations-to-raw-data.sh

# 2. Structure raw BEAT data into 01_structured/<name>/.
uv run python scripts/data/structure_beat.py --config configs/data/remote/beat.yaml

# 3. Tile slides, extract transcripts and cell annotations — one job per sample.
#    Cell annotations are included automatically when cells.parquet is present.
JOB_IDS=()
for SAMPLE_ID in $(uv run python scripts/data/list_samples.py --config configs/data/remote/beat.yaml); do
    JOB_ID=$(sbatch --parsable \
        --cpus-per-task=8 --mem=64G --time=08:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=beat_${SAMPLE_ID} \
        --wrap="uv run python scripts/data/process_beat.py \
            --config configs/data/remote/beat.yaml \
            --filter.include_ids [${SAMPLE_ID}] --filter.exclude_ids null")
    JOB_IDS+=($JOB_ID)
    echo "Submitted ${SAMPLE_ID}: ${JOB_ID}"
done

# 4. Build tile inventory and compute stats after all sample jobs complete.
DEPENDENCY=$(IFS=:; echo "afterok:${JOB_IDS[*]}")
sbatch --dependency=${DEPENDENCY} \
    --cpus-per-task=8 --mem=32G --time=02:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=beat_finalize \
    --wrap="uv run python scripts/data/create_items.py --config configs/data/remote/beat.yaml && \
            uv run python scripts/data/compute_all_items_stats.py --config configs/data/remote/beat.yaml"

# 5. Copy the default gene panel into the managed output directory.
mkdir -p "${DATA_DIR}/03_output/beat/panels/" && cp panels/beat/default.yaml "${DATA_DIR}/03_output/beat/panels/"

# 6. Create filtered item sets and cross-validated splits.
#    expr:            expression task items and splits
#    expr-with-cells: expression task with cell-type features
#    cells:           cell-type prediction items and splits (also used as the canonical split for expr)
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/unil/expr.yaml
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/unil/expr-with-cells.yaml
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/unil/cells.yaml

# 7. Warmup cache (optional — pre-populates tile feature cache before GPU training).
TASK=expression
ITEMS_PATH=cells.json  # cells items/splits are used for both tasks for consistency
PANEL_PATH=default.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"
sbatch \
    --cpus-per-task=10 \
    --mem=32G \
    --time=02:00:00 \
    --output=$HOME/logs/%j.out \
    --wrap="uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/beat/${TASK}/early-fusion.yaml \
        --data.items_path ${ITEMS_PATH} \
        --data.panel_path ${PANEL_PATH} \
        --data.cache_dir=${TASK}/${PANEL_NAME}"

TASK=cell_types
sbatch \
    --cpus-per-task=10 \
    --mem=32G \
    --time=02:00:00 \
    --output=$HOME/logs/%j.out \
    --wrap="uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/beat/${TASK}/early-fusion.yaml \
        --data.items_path ${ITEMS_PATH} \
        --data.panel_path ${PANEL_PATH} \
        --data.cache_dir=${TASK}/${PANEL_NAME}"
```

## Model Training

```bash
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G

TASK=cell_types
#TASK=expression
SPLIT_DIR=cells  # note we only use the cells splits across tasks for consistency
ITEMS_PATH=cells.json  # note we only use the cells items across tasks for consistency
PANEL_PATH=default.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir=null

        # Main run (GPU)
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.items_path ${ITEMS_PATH} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# concat
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true --data.cache_dir=null

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-concat-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.items_path ${ITEMS_PATH} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.fusion_strategy concat \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# freeze_morph_encoder=true
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.freeze_morph_encoder true --debug true --data.cache_dir=null

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-freeze-morph-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.items_path ${ITEMS_PATH} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.freeze_morph_encoder true \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# learnable gate
#for MODEL in early-fusion late-fusion-tile late-fusion-token; do
#  for OUTER in 0 1 2 3; do
#    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
#    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
#    MODEL=early-fusion
#    CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml
#
##    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true --debug true
#
#    sbatch \
#        --cpus-per-task=12 \
#        --mem=${MEMORY} \
#        --gres=gpu:1 \
#        --partition=${PARTITION} \
#        --time=${TIME} \
#        --output=$HOME/logs/%j.out \
#        --job-name=${TASK}-${MODEL}-gate-${OUTER} \
#        --wrap="uv run python scripts/train/supervised.py \
#            --config ${CONFIG} \
#            --data.metadata_path ${METADATA_PATH} \
#            --data.panel_path ${PANEL_PATH} \
#            --backbone.learnable_gate true"
#  done
#done
```

## Evaluation
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/expression.yaml
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/cell_types.yaml
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/expression-cells.yaml
```
