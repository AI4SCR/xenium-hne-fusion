# FM Embeddings Slurm Commands

## BEAT

```bash
for MODEL in UNI UNI2 CONCH_V1.5; do
    sbatch \
        --cpus-per-task=12 \
        --mem=64G \
        --gres=gpu:1 \
        --partition=gpu-l40 \
        --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=fm-emb-${MODEL} \
        --wrap="uv run python scripts/artifacts/create_fm_embeddings.py \
            --config configs/artifacts/beat/unil/fm.yaml \
            --model.name ${MODEL}"
done
```
