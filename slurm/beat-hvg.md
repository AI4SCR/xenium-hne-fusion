# BEAT HVG Slurm Commands

## HVG Panel Creation

```bash
N_TOP_GENES=100
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    SPLIT_DIR=cells
    METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
    PANEL_NAME="hvg-{N_TOP_GENES}/${SPLIT_DIR}/${SPLIT_NAME}"
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/create_panel.py \
            --config configs/artifacts/beat/unil/expr-hvg.yaml \
            --panel.n_top_genes ${N_TOP_GENES} \
            --panel.metadata_path ${METADATA_PATH} \
            --panel.name ${PANEL_NAME}"
done
```

## Cache Warmup

```bash
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"
    PANEL_NAME="${PANEL_PATH%.yaml}"
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=02:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scribble/warmup-cache.py \
            --config configs/train/beat/expression/early-fusion.yaml \
            --items-path expr.json \
            --metadata-path ${METADATA_PATH} \
            --panel-path ${PANEL_PATH} \
            --cache-dir expression/${PANEL_NAME}"
done
```

## Model Training

```bash
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G
TASK=expression

for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"
    PANEL_NAME="${PANEL_PATH%.yaml}"

    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir=null

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-hvg-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# concat
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"
    PANEL_NAME="${PANEL_PATH%.yaml}"

    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true --data.cache_dir=null

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-hvg-concat-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.fusion_strategy concat \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# freeze_morph_encoder=true
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"
    PANEL_NAME="${PANEL_PATH%.yaml}"

    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

    #    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.freeze_morph_encoder true --debug true --data.cache_dir=null

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-hvg-freeze-morph-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.freeze_morph_encoder true \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done
```
