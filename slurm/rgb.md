# BEAT Slurm Commands

## Model Training

```bash
PARTITION=gpu-l40
TIME=12:00:00
MAX_TIME=00:11:00:00
MEMORY=64G

TASK=rgb
SPLIT_DIR=cells  # note we only use the cells splits across tasks for consistency
ITEMS_PATH=cells.json  # note we only use the cells items across tasks for consistency
PANEL_PATH=default.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"

#for OUTER in 0 1 2 3; do
for OUTER in 0; do
    for MODEL in vision expr-token expr-token-vit early-fusion; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir=null
#        uv run python scripts/train/supervised.py \
#                --config ${CONFIG} \
#                --data.items_path ${ITEMS_PATH} \
#                --data.metadata_path ${METADATA_PATH} \
#                --data.panel_path ${PANEL_PATH} \
#                --data.cache_dir=${TASK}/${PANEL_NAME} \
#                --debug=True
#        continue
        
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
                --data.cache_dir=${TASK}/${PANEL_NAME} \
                --trainer.max_time=${MAX_TIME}"
    done
done
```