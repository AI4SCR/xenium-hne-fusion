# BEAT Ray Commands

```bash
# explor cluster
./ray/submit.sh "ls /raid/ray/shared/data/public/silver/xenium-hne-fusion/03_output/beat"
./ray/submit.sh "ls /raid/ray/shared/data/public/silver/xenium-hne-fusion/01_structured/beat/XE_1JAT_01_HNE_1JAT/"
./ray/submit.sh "ls /raid/ray/shared/data/public/silver/xenium-hne-fusion/02_processed/beat/XE_1JAT_01_HNE_1JAT/512_256/1/"

# transfer and process cell annotations, patching the data, now integrated in data processing
./ray/submit.sh "tar --exclude='._*' -xzvf /raid/ray/shared/experimental/tmp/adriano/cell_annotations.tar.gz -C /raid/ray/shared/fmx/data"
./ray/submit.sh "ls /raid/ray/shared/fmx/data/cell_annotations"
./ray/submit.sh "chmod u+x scripts/data/copy-cell-annotations-to-raw-data.sh && scripts/data/copy-cell-annotations-to-raw-data.sh"
./ray/submit.sh "ls /raid/ray/shared/fmx/data/processed-v0/datasets/beat/XE_1JAT_01_HNE_1JAT/"
./ray/submit.sh 'python scripts/data/structure_beat.py --config configs/data/remote/beat.yaml'
./ray/submit.sh 'python scribble/ray_process_beat_cells.py --config configs/data/remote/beat.yaml'
./ray/submit.sh 'python scripts/data/compute_all_items_stats.py --config configs/data/remote/beat.yaml'

# copy default panel
./ray/submit.sh 'mkdir -p "${DATA_DIR}/03_output/beat/panels/" && cp panels/beat/default.yaml "${DATA_DIR}/03_output/beat/panels/"'
./ray/submit.sh 'cp metadata.bak /raid/ray/shared/fmx/data/processed-v0/datasets/beat/metadata.parquet'

# expression artifacts
./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/kaiko/expr.yaml"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_NAME="expr-hvg-${SPLIT_NAME}"
    ./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/kaiko/expr-hvg.yaml --panel.metadata_path ${METADATA_PATH} --panel.name ${PANEL_NAME}"
done

# cell type artifacts
./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/kaiko/cells.yaml"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="cells/${SPLIT_NAME}.parquet"
    PANEL_NAME="cells-hvg-${SPLIT_NAME}"
    ./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/beat/kaiko/cells-hvg.yaml --panel.metadata_path ${METADATA_PATH} --panel.name ${PANEL_NAME}"
done

# expression training
TASK=expression
ITEMS_PATH="expr.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
      CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"
#      ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH}"
      ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true"
    done
done

# cell type training
TASK=cell_types
ITEMS_PATH="cells.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="cells/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="cells-hvg-${SPLIT_NAME}.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
      CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"
      ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH}"
#      ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true"
    done
done

# expression concat fusion
TASK=expression
ITEMS_PATH="expr.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"
#        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat"
        ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true"
    done
done

# cell type concat fusion
TASK=cell_types
ITEMS_PATH="cells.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="cells/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="cells-hvg-${SPLIT_NAME}.yaml"

    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat"
#        ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.fusion_strategy concat --debug true"
    done
done

# expression learnable gate
TASK=expression
MODEL=early-fusion
ITEMS_PATH="expr.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="expr/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="expr-hvg-${SPLIT_NAME}.yaml"
    CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"

#    ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true"
    ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true --debug true"
done

# cell type learnable gate
TASK=cell_types
MODEL=early-fusion
ITEMS_PATH="cells.json"
for OUTER in 0 1 2 3; do
    SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
    METADATA_PATH="cells/${SPLIT_NAME}.parquet"
    PANEL_PATH="default.yaml"
#    PANEL_PATH="cells-hvg-${SPLIT_NAME}.yaml"
    CONFIG="configs/train/beat/${TASK}/${MODEL}.yaml"

    ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true"
#    ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config ${CONFIG} --data.items_path ${ITEMS_PATH} --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --backbone.learnable_gate true --debug true"
done

# data processing
./ray/submit.sh "python scripts/data/run_beat.py --config configs/data/remote/beat.yaml --executor ray --stage all --overwrite true"
```
