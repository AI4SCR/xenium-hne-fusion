# Owkin Slurm Commands


## Data Preparation

Steps must run in order; each stage depends on the previous one.

```bash
# 1. Transfer cell annotations into the raw data normalize_cell_type_categoriesdirectory.

# 2. Structure raw OWKIN data into 01_structured/<name>/.
uv run python scripts/data/structure_owkin.py --config configs/data/remote/owkin.yaml

uv run python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/scribble/convert-owkin-to-pyramidal.py

# 3. Tile slides, extract transcripts and cell annotations — one job per sample.
#    Cell annotations are included automatically when cells.parquet is present.
JOB_IDS=()
#for SAMPLE_ID in $(uv run python scripts/data/list_samples.py --config configs/data/remote/owkin.yaml); do
for SAMPLE_ID in CH_C_518a_x2  CH_C_525a_x2  CH_C_527a_x2  CH_D_529a_x2  CH_D_531a_x2  CH_G_532a_x2  CH_G_534a_x1  CH_G_535a_xr  CH_C_523a_x2  CH_C_526a_x1  CH_C_527a_xr  CH_D_530a_x2  CH_D_531a_xr  CH_G_533a_x2  CH_G_535a_x2  CH_G_536a_x2; do
    JOB_ID=$(sbatch --parsable \
        --cpus-per-task=8 --mem=128G --time=06:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=beat_${SAMPLE_ID} \
        --wrap="uv run python scripts/data/process_owkin.py \
            --config configs/data/remote/owkin.yaml \
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
    --wrap="uv run python scripts/data/create_items.py --config configs/data/remote/owkin.yaml && \
            uv run python scripts/data/compute_all_items_stats.py --config configs/data/remote/owkin.yaml --cell-type-col first_type"

# 5. Copy the default gene panel into the managed output directory.
mkdir -p "${DATA_DIR}/03_output/owkin/panels/" && cp panels/owkin/owkin-beat.yaml "${DATA_DIR}/03_output/owkin/panels"

# 6. Create filtered item sets and cross-validated splits.
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/owkin/cells.yaml

# 7. Warmup cache (optional — pre-populates tile feature cache before GPU training).
TASK=proteins
ITEMS_PATH=cells.json  # cells items/splits are used for both tasks for consistency
PANEL_PATH=owkin-beat.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"
sbatch \
    --cpus-per-task=10 \
    --mem=32G \
    --time=02:00:00 \
    --output=$HOME/logs/%j.out \
    --wrap="uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/owkin/${TASK}/early-fusion.yaml \
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
TIME=09:00:00
MAX_TIME=00:08:00:00
MEMORY=64G

TASK=proteins
SPLIT_DIR=cells  # note we only use the cells splits across tasks for consistency
ITEMS_PATH=cells.json  # cells items/splits are used for both tasks for consistency
PANEL_PATH=owkin-beat.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"
#for OUTER in 0 1 2 3; do
for OUTER in 0; do
    for MODEL in early-fusion expr-token-vit late-fusion-tile vision; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/owkin/${TASK}/${MODEL}.yaml
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir=null
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir="${TASK}/${PANEL_NAME}"
#        break
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
                --trainer.max_time ${MAX_TIME} \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done

# concat
for OUTER in 0; do
    for MODEL in early-fusion late-fusion-tile; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/owkin/${TASK}/${MODEL}.yaml

        # Main run (GPU)
        sbatch \
            --cpus-per-task=12 \
            --mem=${MEMORY} \
            --gres=gpu:1 \
            --partition=${PARTITION} \
            --time=${TIME} \
            --output=$HOME/logs/%j.out \
            --job-name=${TASK}-${MODEL}-${OUTER}-concat \
            --wrap="uv run python scripts/train/supervised.py \
                --config ${CONFIG} \
                --data.items_path ${ITEMS_PATH} \
                --data.metadata_path ${METADATA_PATH} \
                --data.panel_path ${PANEL_PATH} \
                --trainer.max_time ${MAX_TIME} \
                --data.cache_dir=${TASK}/${PANEL_NAME} \
                --backbone.fusion_strategy concat"
    done
done

```

```bash
PARTITION=gpu-l40
TIME=09:00:00
MAX_TIME=00:08:00:00
MEMORY=64G

TASK=proteins
SPLIT_DIR=cells  # note we only use the cells splits across  nexttasks for consistency
ITEMS_PATH=cells.json  # cells items/splits are used for both tasks for consistency
PANEL_PATH=owkin-beat.yaml
PANEL_NAME="${PANEL_PATH%.yaml}"
#for OUTER in 0 1 2 3; do
for OUTER in 0; do
    for MODEL in early-fusion expr-token-vit late-fusion-tile vision; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        METADATA_PATH="${SPLIT_DIR}/${SPLIT_NAME}.parquet"
        CONFIG=configs/train/owkin/proteins-regularized/${MODEL}.yaml
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir=null
#        uv run python scripts/train/supervised.py --config ${CONFIG} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true --data.cache_dir="${TASK}/${PANEL_NAME}"
#        break
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
                --trainer.max_time ${MAX_TIME} \
                --data.cache_dir=${TASK}/${PANEL_NAME}"
    done
done
```