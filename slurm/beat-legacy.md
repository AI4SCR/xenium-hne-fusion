# BEAT Legacy Slurm Commands

```bash
# --- legacy splits (cells-legacy: same sample→split as legacy codebase, sourced from cells.json) ---
# First regenerate the splits:
uv run python issues/create-legacy-splits/create_legacy_splits.py

# cell_types training on legacy splits
TASK=cell_types
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="cells-legacy/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.items_path cells.json --debug true

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-legacy-${MODEL}-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.items_path cells.json"
    done
done

# eval plots
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/cell_types-legacy.yaml
```
