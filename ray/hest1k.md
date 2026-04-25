# HEST1K Ray Commands

```bash
# data
./ray/submit.sh "python scripts/data/run_hest1k.py --config configs/data/remote/hest1k.yaml --executor ray --stage all --overwrite true"

# artifacts
for ORGAN in breast lung pancreas bowel; do

    ./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/hest1k/${ORGAN}.yaml"

    for OUTER in 0 1 2 3; do
        SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
        echo "Creating panel for ${ORGAN} and ${SPLIT_NAME}"

        cmd="python scripts/artifacts/create_panel.py \
            --config configs/artifacts/hest1k/${ORGAN}.yaml \
            --panel.metadata_path ${ORGAN}/${SPLIT_NAME}.parquet \
            --panel.name ${ORGAN}-hvg-${SPLIT_NAME}"

        echo "Running: ${cmd}"
        ./ray/submit.sh "${cmd}"
    done
done

# training
TASK=expression
for ORGAN in breast lung pancreas bowel; do
  for OUTER in 0 1 2 3; do
    METADATA_PATH="${ORGAN}/outer=${OUTER}-inner=0-seed=0.parquet"
    PANEL_PATH="${ORGAN}-hvg-outer=${OUTER}-inner=0-seed=0.yaml"
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
#      ./ray/submit.sh --entrypoint-num-gpus 0 --entrypoint-num-cpus 2 "python scripts/train/supervised.py --config configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH} --debug true"
      ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 "python scripts/train/supervised.py --config configs/train/hest1k/${TASK}/${ORGAN}/${MODEL}.yaml --data.metadata_path ${METADATA_PATH} --data.panel_path ${PANEL_PATH}"
    done
  done
done
```
