# FM Embeddings Slurm Commands

## Install

Create an environment compatible with CONCH_v1.5

```bash
uv venv --python 3.11 .venv-titan
source .venv-titan/bin/activate
uv pip install \
    lazyslide \
    torch==2.7 \
    transformers==4.51.3 \
    einops \
    einops-exts \
    huggingface-hub \
    python-dotenv \
    pandas \
    jsonargparse \
    loguru \
    tqdm
uv pip install ../ai4bmr-learn --no-deps
uv pip install -e . --no-deps
```

## BEAT

```bash
#for MODEL in uni; do
for MODEL in uni uni2 conch_v1.5; do
    sbatch \
        --cpus-per-task=12 \
        --mem=64G \
        --gres=gpu:1 \
        --partition=gpu-gh \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=fm-emb-${MODEL} \
        --wrap=".venv-titan/bin/python scripts/artifacts/create_fm_embeddings.py \
            --config configs/artifacts/beat/unil/fm.yaml \
            --model.name ${MODEL}"
done
```
