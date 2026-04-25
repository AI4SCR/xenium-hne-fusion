# HEST1K Slurm Commands

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
