# BEAT Slurm Commands

```bash
# NOTE: transfer and process cell annotations
uv run python chmod u+x scripts/data/copy-cell-annotations-to-raw-data.sh && scripts/data/copy-cell-annotations-to-raw-data.sh
uv run python scripts/data/structure_beat.py --config configs/data/remote/beat.yaml
./scribble/sbatch_process_beat_cells.sh --config configs/data/remote/beat.yaml
uv run scripts/data/compute_all_items_stats.py --config configs/data/remote/beat.yaml

# copy default panel
uv run mkdir -p "${DATA_DIR}/03_output/beat/panels/" && cp panels/beat/default.yaml "${DATA_DIR}/03_output/beat/panels/"

uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/unil/expr.yaml
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/unil/cells.yaml

# hvg panels (run after artifact jobs)
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_NAME="expr-hvg-${SPLIT_NAME}"
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/create_panel.py \
            --config configs/artifacts/beat/unil/expr-hvg.yaml \
            --panel.metadata_path ${METADATA_PATH} \
            --panel.name ${PANEL_NAME}"
done

# warmup cache
for OUTER in 0 1 2 3; do
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=02:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scribble/warmup-cache.py --outer ${OUTER}"
done

# TASK=cell_types
TASK=expression
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G
#PANEL_PATH=default.yaml
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="expr/${SPLIT_NAME}.parquet"
        PANEL_PATH="expr-hvg-outer=${OUTER}-inner=0-seed=0.yaml"
        PANEL_NAME="${PANEL_PATH%.yaml}"
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
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --data.cache_dir=expression/${PANEL_NAME}"
    done
done

# concat
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="expr/${SPLIT_NAME}.parquet"
        PANEL_PATH="expr-hvg-outer=${OUTER}-inner=0-seed=0.yaml"
        PANEL_NAME="${PANEL_PATH%.yaml}"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true

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
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.fusion_strategy concat \
                --data.cache_dir=expression/${PANEL_NAME}"
    done
done

# freeze_morph_encoder=true
for OUTER in 0 1 2 3; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="expr/${SPLIT_NAME}.parquet"
        PANEL_PATH="expr-hvg-outer=${OUTER}-inner=0-seed=0.yaml"
        PANEL_NAME="${PANEL_PATH%.yaml}"
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true

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
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.freeze_morph_encoder true \
                --data.cache_dir=expression/${PANEL_NAME}"
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

# eval plots
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/expression.yaml
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/beat/cell_types.yaml
```
