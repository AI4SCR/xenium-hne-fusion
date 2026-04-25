# HESCAPE Ray Commands

```bash
# artifacts
for PANEL in \
  "breast human-breast-panel" \
  "lung-healthy human-lung-healthy-panel" \
  "bowel human-colon-panel" \
  "human-5k human-5k-panel" \
  "human-immuno-oncology human-immuno-oncology-panel" \
  "human-multi-tissue human-multi-tissue-panel"; do
  ./ray/submit.sh "python scripts/artifacts/create_artifacts.py --config configs/artifacts/hescape/${ARTIFACT_NAME}.yaml"
done

# fixed splits
./ray/submit.sh "python scripts/artifacts/create_hescape_splits.py"

# training — base models
TASK=expression
ORGANS=(breast bowel lung-healthy human-immuno-oncology human-multi-tissue)

for ORGAN in "${ORGANS[@]}"; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision expr-tile expr-token; do
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
            "python scripts/train/supervised.py \
                --config configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml"
    done
done

# concat fusion
for ORGAN in "${ORGANS[@]}"; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
            "python scripts/train/supervised.py \
                --config configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml \
                --backbone.fusion_strategy concat"
    done
done

# learnable gate
for ORGAN in "${ORGANS[@]}"; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token; do
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
            "python scripts/train/supervised.py \
                --config configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml \
                --backbone.learnable_gate true"
    done
done

# freeze morph
FREEZE_MORPH=true
for ORGAN in "${ORGANS[@]}"; do
    for MODEL in early-fusion late-fusion-tile late-fusion-token vision; do
        ./ray/submit.sh --entrypoint-num-gpus 1 --entrypoint-num-cpus 12 \
            "python scripts/train/supervised.py \
                --config configs/train/hescape/${TASK}/${ORGAN}/${MODEL}.yaml \
                --backbone.freeze_morph ${FREEZE_MORPH}"
    done
done
```
