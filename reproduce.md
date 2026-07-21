# Reproduce v0 results

Warm the tile cache against the `v0` data version (`DATA_DIR=xenium-hne-fusion-v0`), using the
`cells` item set / split / panel from that version, so training can reuse a pre-populated cache
under `cache_dir=v0_cells`.

- items: `03_output/owkin/items/cells.json`
- split: `03_output/owkin/splits/cells/outer=0-inner=0-seed=0.parquet`
- panel: `03_output/owkin/panels/owkin-beat.yaml`
- cache: `03_output/owkin/cache/v0_cells`

```bash
uv run python scripts/artifacts/warmup_cache.py \
    --config configs/train/owkin/proteins/early-fusion.yaml \
    --train.data.data_dir /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0 \
    --train.data.name owkin \
    --train.data.items_path cells.json \
    --train.data.metadata_path cells/outer=0-inner=0-seed=0.parquet \
    --train.data.panel_path owkin-beat.yaml \
    --train.data.cache_dir v0_cells
```

## Train protein-prediction models (v0)

`cells/outer=<N>-inner=0-seed=0.parquet` matches the original v0 run configs (commit `a5fd135`),
which fixed `inner=0`. The `cells` split has 4 outer folds (0-3, see
`03_output/owkin/splits/cells/`). The original v0 sweep only had 4 run configs — no `expr-resmlp`
yet (added later, commit `10aae7f`) — so reproduce with `vision`, `expr-token-vit`,
`late-fusion-tile`, `early-fusion` only:

```bash
for CONFIG in vision expr-token-vit late-fusion-tile early-fusion; do
    for OUTER in 0 1 2 3; do
        uv run python scripts/train/supervised.py \
            --config configs/train/owkin/proteins/${CONFIG}.yaml \
            --debug=True \
            --train.data.data_dir /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0 \
            --train.data.name owkin \
            --train.data.items_path cells.json \
            --train.data.metadata_path cells/outer=${OUTER}-inner=0-seed=0.parquet \
            --train.data.panel_path owkin-beat.yaml \
            --train.data.cache_dir v0_cells \
            --train.wandb.project xe-hne-fus-protein-v0 \
            --train.wandb.tags "[owkin, v0, reproduce, c_d_cells]"
    done
done
```

Submit one dedicated sbatch loop, sweeping `CONFIG` and every outer fold (0-3):

```bash
PARTITION=gpu-l40
for CONFIG in vision expr-token-vit late-fusion-tile early-fusion; do
    for OUTER in 0 1 2 3; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=10 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_reproduce_v0_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.data_dir /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0 \
                --train.data.name owkin \
                --train.data.items_path cells.json \
                --train.data.metadata_path cells/outer=${OUTER}-inner=0-seed=0.parquet \
                --train.data.panel_path owkin-beat.yaml \
                --train.data.cache_dir v0_cells \
                --train.wandb.project xe-hne-fus-protein-v0 \
                --train.wandb.tags [owkin,v0,reproduce,c_d_cells]"
    done
done
```
