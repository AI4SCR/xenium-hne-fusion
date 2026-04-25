# HESCAPE Slurm Commands

```bash
ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)
MODELS=(early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token)
MODELS=(expr-tile expr-token)

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

ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)
MODELS=(early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token)
MODELS=(expr-tile expr-token)

# base models
for OUTER in 0 1 2 3 4; do
  for ORGAN in "${ORGANS[@]}"; do
    METADATA_PATH="hescape/${ORGAN}/outer=${OUTER}-seed=0.parquet"
    PANEL_PATH="hescape/${ORGAN}.yaml"
    for MODEL in "${MODELS[@]}"; do
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


# eval plots
ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)
ORGAN=breast
uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/hescape/${ORGAN}.yaml
for ORGAN in "${ORGANS[@]}"; do
    uv run python scripts/eval/plot_wandb_scores.py --config configs/eval/hescape/${ORGAN}.yaml
done

# paired t-test stats vs the eval config baseline
ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)
ORGAN=breast
uv run python scripts/eval/paired_t_tests.py --config configs/eval/hescape/${ORGAN}.yaml
for ORGAN in "${ORGANS[@]}"; do
    uv run python scripts/eval/paired_t_tests.py --config configs/eval/hescape/${ORGAN}.yaml
done
```
