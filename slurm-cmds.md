# Slurm Commands

## HEST1K

```bash
uv run python scripts/data/create_items.py --config configs/data/remote/hest1k.yaml
uv run python scripts/data/compute_all_items_stats.py --config configs/data/remote/hest1k.yaml

for ORGAN in bowel breast lung pancreas; do
    sbatch \
        --cpus-per-task=4 \
        --mem=32G \
        --time=01:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/filter_items.py --config configs/artifacts/hest1k/${ORGAN}.yaml"
done

for ORGAN in bowel breast lung pancreas; do
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=01:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/compute_items_stats.py --config configs/artifacts/hest1k/${ORGAN}.yaml"
done

for ORGAN in bowel breast lung pancreas; do
    sbatch \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/report_feature_overlap.py --config configs/artifacts/hest1k/${ORGAN}.yaml"
done

for ORGAN in bowel breast lung pancreas; do
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/hest1k/${ORGAN}.yaml"
done

# Run after the artifact jobs have finished.
for ORGAN in bowel breast lung pancreas; do
    for OUTER in 1 2 3; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${ORGAN}/${SPLIT_NAME}.parquet"
        PANEL_NAME="${ORGAN}-hvg-${SPLIT_NAME}"
        echo "Creating panel ${PANEL_NAME} from split ${METADATA_PATH}"

        sbatch \
            --cpus-per-task=10 \
            --mem=32G \
            --time=04:00:00 \
            --output=~/logs/%j.out \
            --wrap="uv run python scripts/artifacts/create_panel.py --config configs/artifacts/hest1k/${ORGAN}.yaml --panel.metadata_path ${METADATA_PATH} --panel.name ${PANEL_NAME}"
    done
done

# training
PARTITION=gpu-l40
TIME=04:00:00
MEMORY=64G
TASK=expression
#OUTER=0
#ORGAN=lung
#MODEL=expr-token
for OUTER in 0 1 2 3; do
  for ORGAN in bowel breast lung pancreas human-immuno-oncology human-multi-tissue; do
    METADATA_PATH="${ORGAN}/outer=${OUTER}-inner=0-seed=0.parquet"
    PANEL_PATH="${ORGAN}-hvg-outer=${OUTER}-inner=0-seed=0.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        CONFIG=configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml
        
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true

        # Main run (GPU)
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${ORGAN}-${TASK}-${MODEL}-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH}"
    done
  done
done

# concat
TASK=expression
for OUTER in 0 1 2 3; do
  for ORGAN in bowel breast lung pancreas; do
    METADATA_PATH="${ORGAN}/outer=${OUTER}-inner=0-seed=0.parquet"
    PANEL_PATH="${ORGAN}-hvg-outer=${OUTER}-inner=0-seed=0.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG=configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml

#            uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${ORGAN}-${TASK}-${MODEL}-concat-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.fusion_strategy concat"
    done
  done
done

# learnable gate
TASK=expression
for OUTER in 0 1 2 3; do
  for ORGAN in bowel breast lung pancreas; do
    METADATA_PATH="${ORGAN}/outer=${OUTER}-inner=0-seed=0.parquet"
    PANEL_PATH="${ORGAN}-hvg-outer=${OUTER}-inner=0-seed=0.yaml"
    
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG=configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml

#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true --debug true

        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${ORGAN}-${TASK}-${MODEL}-gate-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.learnable_gate true"
    done
  done
done
```

## BEAT

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

# TASK=cell_types
TASK=expression
PARTITION=gpu-l40
TIME=04:30:00
MEMORY=64G
PANEL_PATH=default.yaml
for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
  for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"

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
            --data.panel_path ${PANEL_PATH}"
  done
done

# concat
for MODEL in early-fusion late-fusion-tile late-fusion-token; do
  for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
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
            --backbone.fusion_strategy concat"
  done
done

# learnable gate
for MODEL in early-fusion late-fusion-tile late-fusion-token; do
  for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    MODEL=early-fusion
    CONFIG=configs/train/beat/${TASK}/${MODEL}.yaml

#    uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true --debug true

    sbatch \
        --cpus-per-task=12 \
        --mem=${MEMORY} \
        --gres=gpu:1 \
        --partition=${PARTITION} \
        --time=${TIME} \
        --output=$HOME/logs/%j.out \
        --job-name=${TASK}-${MODEL}-gate-${OUTER} \
        --wrap="uv run python scripts/train/supervised.py \
            --config ${CONFIG} \
            --data.metadata_path ${METADATA_PATH} \
            --data.panel_path ${PANEL_PATH} \
            --backbone.learnable_gate true"
  done
done

# freeze morph
FREEZE_MORPH=true
for ORGAN in "${ORGANS[@]}"; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
            "python scripts/train/supervised.py \
                --config configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml \
                --backbone.freeze_morph ${FREEZE_MORPH}"
    done
done

```

## HESCAPE

```bash
ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)

# artifacts (filter items + cross-validated splits + compute stats)
for ORGAN in "${ORGANS[@]}"; do
    sbatch \
        --cpus-per-task=10 \
        --mem=32G \
        --time=01:00:00 \
        --output=$HOME/logs/%j.out \
        --wrap="uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/hescape/${ORGAN}.yaml"
done

# fixed hescape splits (all organs in one call)
uv run python scripts/artifacts/create_hescape_splits.py

# hvg panels (run after hescape splits are created)
uv run python scripts/artifacts/create_hescape_panels.py

# training
TASK=expression
PARTITION=gpu-l40
#PARTITION=gpu-gh
TIME=04:00:00
MEMORY=64G
MAX_EPOCHS=50
FREEZE_MORPH=true

# base models
for OUTER in 0 1 2 3 4; do
  for ORGAN in "${ORGANS[@]}"; do
    METADATA_PATH="hescape/${ORGAN}/outer=${OUTER}-seed=0.parquet"
    PANEL_PATH="hescape/${ORGAN}.yaml"
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        CONFIG=configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug=true --data.cache_dir=null
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=hescape-${ORGAN}-${MODEL}-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --trainer.max_epochs ${MAX_EPOCHS}"
    done
  done
done

# concat fusion
for OUTER in 0 1 2 3 4; do
  for ORGAN in "${ORGANS[@]}"; do
    METADATA_PATH="hescape/${ORGAN}/outer=${OUTER}-seed=0.parquet"
    PANEL_PATH="hescape/${ORGAN}.yaml"
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG=configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=hescape-${ORGAN}-${MODEL}-concat-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.fusion_strategy concat \
                --trainer.max_epochs ${MAX_EPOCHS}"
    done
  done
done

# learnable gate
for OUTER in 0 1 2 3 4; do
  for ORGAN in "${ORGANS[@]}"; do
    METADATA_PATH="hescape/${ORGAN}/outer=${OUTER}-seed=0.parquet"
    PANEL_PATH="hescape/${ORGAN}.yaml"
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG=configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=hescape-${ORGAN}-${MODEL}-gate-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.learnable_gate true \
                --trainer.max_epochs ${MAX_EPOCHS}"
    done
  done
done

# freeze morph
for OUTER in 0 1 2 3 4; do
  for ORGAN in "${ORGANS[@]}"; do
    METADATA_PATH="hescape/${ORGAN}/outer=${OUTER}-seed=0.parquet"
    PANEL_PATH="hescape/${ORGAN}.yaml"
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        CONFIG=configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=hescape-${ORGAN}-${MODEL}-freeze-morph-${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --backbone.freeze_morph ${FREEZE_MORPH} \
                --trainer.max_epochs ${MAX_EPOCHS}"
    done
  done
done
```
