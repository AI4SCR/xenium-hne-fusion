# Runs

## owkin

Prerequisite for `process.py` (pyramidalizes `wsi.tiff` via `pyvips`): `libvips` must be on
`LD_LIBRARY_PATH`, set in your shell profile (not `.env` — dotenv loads too late to affect
`dlopen`'s search path):

```bash
export LD_LIBRARY_PATH=/work/FAC/FBM/DBC/mrapsoma/prometex/envs/adrianom/libvips-only/lib:$LD_LIBRARY_PATH
```

Structure raw samples into `01_structured/`:

```bash
uv run python scripts/data/owkin/structure.py --config configs/data/owkin.yaml
```

Process structured samples into `02_processed/` (tiling, transcript/cell/protein extraction):

```bash
uv run python scripts/data/owkin/process.py --config configs/data/owkin.yaml
```

Both are idempotent — rerunning skips already-structured/already-processed samples unless
`--overwrite true` is passed.

### Process one sample per job

Test a single sample locally before submitting the full loop as SLURM jobs:

```bash
uv run python scripts/data/owkin/process.py \
    --config configs/data/owkin.yaml \
    --data.filter.include_ids="[CH_C_518a_x2]"
```
For SLURM array jobs, restrict a single run to one sample via `--data.filter.include_ids`. 
Submit the full loop as one SLURM job per sample:

```bash
for SAMPLE_ID in CH_C_518a_x2 CH_C_523a_x2 CH_C_525a_x2 CH_C_526a_x1 CH_C_527a_x2 CH_C_527a_xr \
    CH_D_529a_x2 CH_D_530a_x2 CH_D_531a_x2 CH_D_531a_xr CH_G_532a_x2 CH_G_533a_x2 CH_G_534a_x1 \
    CH_G_535a_x2 CH_G_535a_xr CH_G_536a_x2; do
    sbatch \
        --account=rgottar1_spatial \
        --cpus-per-task=4 --mem=128G --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=owkin_${SAMPLE_ID} \
        --wrap="uv run python scripts/data/owkin/process.py \
            --config configs/data/owkin.yaml \
            --data.filter.include_ids=[${SAMPLE_ID}]"
done
```

### Create the source items list (`items/all.json`)

Collects every tile (`tile.pt`) written by `process.py` under `02_processed/owkin/<sample_id>/512_256/`
into `items/all.json` — no transcript/expression file required, so empty tiles are included too.
`build_items.py` only reads `data_dir`/`tile_px`/`stride_px`/`cell_type_col` from the config and
always writes to the hardcoded `items/all.json` (never `items.name`), so any of the per-organ-group
configs below works — it does not filter or produce a named item set:

```bash
uv run python scripts/artifacts/build_items.py --config configs/artifacts/owkin/c_cells.yaml
```

`build_items.py` also computes stats on `all.json` by default (`--compute-stats true`), writing
`statistics/all.parquet`; pass `--compute-stats false` to skip that (e.g. when only re-tiling and
stats already exist).

### Compute stats for a filtered subset

After `filter_items.py` produces a filtered `items/<name>.json` (e.g. `c_cells.json`, using the
thresholds under `artifacts.items.filter` in `c_cells.yaml` against the default stats above), run
`compute_items_stats.py` with the same config — it reads `items/${artifacts.items.name}.json`
(`items.name` in `c_cells.yaml` is `c_cells`), so no `--items-path` override is needed:

```bash
uv run python scripts/artifacts/compute_items_stats.py \
    --config configs/artifacts/owkin/c_cells.yaml
```

### Create filtered artifacts (c_cells / d_cells / g_cells)

Each per-organ-group config goes through the same four stages. `items/all.json` and
`statistics/all.parquet` must already exist first (see `build_items.py` above):

```bash
for NAME in c_cells d_cells g_cells c_d_cells; do
    uv run python scripts/artifacts/filter_items.py --config configs/artifacts/owkin/${NAME}.yaml
    uv run python scripts/artifacts/build_splits.py --config configs/artifacts/owkin/${NAME}.yaml
    uv run python scripts/artifacts/build_panel.py --config configs/artifacts/owkin/${NAME}.yaml
    uv run python scripts/artifacts/compute_items_stats.py \
        --config configs/artifacts/owkin/${NAME}.yaml
done
```

- `filter_items.py` applies the thresholds under `artifacts.items.filter` in `${NAME}.yaml` to
  `items/all.json` and writes `items/${NAME}.json`.
- `build_splits.py` reads sample→split assignments from `splits/owkin.yaml` (hand-authored,
  checked into the repo) and joins them onto the filtered items by `sample_id` — no GroupKFold,
  no `test_size`/`val_size` tuning. For these item-sets `split.name` matches `items.name`, and
  `splits/owkin.yaml` must have an entry for that name (a list of `{train, val, test}` sample-id
  folds, materialized to `splits/${NAME}/outer=<i>.parquet`).
- `build_panel.py` intersects `feature_universe.txt` across the sample IDs present in
  `items/${NAME}.json` and writes the result as `source_panel` to
  `DATA_DIR/03_output/owkin/panels/${NAME}.yaml` (empty `target_panel`).
- `compute_items_stats.py` recomputes stats/figures for the filtered item set.

Submit the same four stages as one SLURM job per item-set (each job runs its stages serially,
one job per `NAME`):

```bash
for NAME in c_cells d_cells g_cells; do
    sbatch \
        --account=rgottar1_spatial \
        --cpus-per-task=10 --mem=32G --time=04:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=owkin_artifacts_${NAME} \
        --wrap="uv run python scripts/artifacts/filter_items.py --config configs/artifacts/owkin/${NAME}.yaml && \
            uv run python scripts/artifacts/build_splits.py --config configs/artifacts/owkin/${NAME}.yaml && \
            uv run python scripts/artifacts/build_panel.py --config configs/artifacts/owkin/${NAME}.yaml && \
            uv run python scripts/artifacts/compute_items_stats.py --config configs/artifacts/owkin/${NAME}.yaml"
done
```

### Create cross-organ splits (c_on_d / d_on_c)

`c_on_d` and `d_on_c` both filter to the same 8 samples (5 C + 3 D) and share one item set,
`items/c_d_cells.json` — only the train/val/test *direction* differs between them (C trains and
tests on held-out D, D trains and tests on held-out C). Because of that, `items.name` (`c_d_cells`)
no longer matches `split.name` (`c_on_d` / `d_on_c`) in `configs/artifacts/owkin/{c_on_d,d_on_c}.yaml`
— both configs point at the same `items.name`/`panel.name: c_d_cells`, only `split.name` differs.

`filter_items.py` / `build_panel.py` / `compute_items_stats.py` only need to run once against the
shared item set (either config works, since `items.name` is identical); `build_splits.py` must run
once per split direction:

```bash
uv run python scripts/artifacts/filter_items.py --config configs/artifacts/owkin/c_on_d.yaml
uv run python scripts/artifacts/build_panel.py --config configs/artifacts/owkin/c_on_d.yaml
uv run python scripts/artifacts/compute_items_stats.py --config configs/artifacts/owkin/c_on_d.yaml

for SPLIT in c_on_d d_on_c; do
    uv run python scripts/artifacts/build_splits.py --config configs/artifacts/owkin/${SPLIT}.yaml
done
```

Submit as one SLURM job (shared item/panel/stats stages run once, both split directions built
after):

```bash
sbatch \
    --account=rgottar1_spatial \
    --cpus-per-task=10 --mem=32G --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=owkin_artifacts_c_d_cells \
    --wrap="uv run python scripts/artifacts/filter_items.py --config configs/artifacts/owkin/c_on_d.yaml && \
        uv run python scripts/artifacts/build_panel.py --config configs/artifacts/owkin/c_on_d.yaml && \
        uv run python scripts/artifacts/compute_items_stats.py --config configs/artifacts/owkin/c_on_d.yaml && \
        uv run python scripts/artifacts/build_splits.py --config configs/artifacts/owkin/c_on_d.yaml && \
        uv run python scripts/artifacts/build_splits.py --config configs/artifacts/owkin/d_on_c.yaml"
```

### Warm the tile cache

`warmup_cache.py` populates `DATA_DIR/03_output/owkin/cache/protein/<ITEMS>` before
training so the first epoch doesn't pay tile-read cost. `ITEMS` selects which item
set/panel to warm — must match one of the artifacts configs above (`c_cells`,
`d_cells`, `g_cells`):

```bash
for ITEMS in c_cells d_cells g_cells; do
    uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/owkin/proteins/early-fusion.yaml \
        --train.data.items_path ${ITEMS}.json \
        --train.data.metadata_path ${ITEMS}/outer=0.parquet \
        --train.data.panel_path ${ITEMS}.yaml \
        --train.data.cache_dir protein/${ITEMS}
done
```

Submit the same warmup as one SLURM job per item-set:

```bash
for ITEMS in c_cells d_cells g_cells; do
    sbatch \
        --account=rgottar1_spatial \
        --cpus-per-task=10 --mem=32G --time=02:00:00 \
        --output=$HOME/logs/%j.out \
        --job-name=owkin_warmup_${ITEMS} \
        --wrap="uv run python scripts/artifacts/warmup_cache.py \
            --config configs/train/owkin/proteins/early-fusion.yaml \
            --train.data.items_path ${ITEMS}.json \
            --train.data.metadata_path ${ITEMS}/outer=0.parquet \
            --train.data.panel_path ${ITEMS}.yaml \
            --train.data.cache_dir protein/${ITEMS}"
done
```

For `c_on_d`/`d_on_c`, `items_path`/`panel_path`/`cache_dir` all stay `c_d_cells` (shared item
set and cache — the underlying tiles are identical either direction); only `metadata_path`
varies with the split direction, so warm the cache once and reuse it for both:

```bash
uv run python scripts/artifacts/warmup_cache.py \
    --config configs/train/owkin/proteins/early-fusion.yaml \
    --train.data.items_path c_d_cells.json \
    --train.data.metadata_path c_on_d/outer=0.parquet \
    --train.data.panel_path c_d_cells.yaml \
    --train.data.cache_dir protein/c_d_cells
```

Submit as one SLURM job:

```bash
sbatch \
    --account=rgottar1_spatial \
    --cpus-per-task=10 --mem=32G --time=02:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=owkin_warmup_c_d_cells \
    --wrap="uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/owkin/proteins/early-fusion.yaml \
        --train.data.items_path c_d_cells.json \
        --train.data.metadata_path c_on_d/outer=0.parquet \
        --train.data.panel_path c_d_cells.yaml \
        --train.data.cache_dir protein/c_d_cells"
```

### Train protein-prediction models

Each config under `configs/train/owkin/proteins/` (`early-fusion`, `expr-token-vit`,
`expr-resmlp`, `late-fusion-tile`, `vision`) fixes an architecture; override `--train.data.*` to pick
the item-set/panel/cache to train on. Keep `ITEMS` consistent with whichever cache you
warmed above. Each item-set's split has a fixed number of outer folds (`splits/${NAME}/outer=<i>.parquet`,
`i` from `splits/owkin.yaml`): `c_cells` and `g_cells` have 5 folds (0-4), `d_cells` has 3 (0-2).

```bash
for ITEMS in c_cells d_cells g_cells; do
    for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
        uv run python scripts/train/supervised.py \
            --config configs/train/owkin/proteins/${CONFIG}.yaml \
            --debug=True \
            --train.data.items_path ${ITEMS}.json \
            --train.data.metadata_path ${ITEMS}/outer=0.parquet \
            --train.data.panel_path ${ITEMS}.yaml \
            --train.data.cache_dir protein/${ITEMS} \
            --train.wandb.tags "[owkin, ${ITEMS}]"
    done
done
```

Submit one dedicated sbatch loop per item-set, sweeping `CONFIG` and every outer fold `OUTER` that
item-set actually has (each item-set gets its own `(CONFIG, OUTER)` sweep since fold counts differ):

```bash
PARTITION=gpu-l40
for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2 3 4; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_c_cells_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path c_cells.json \
                --train.data.metadata_path c_cells/outer=${OUTER}.parquet \
                --train.data.panel_path c_cells.yaml \
                --train.data.cache_dir protein/c_cells \
                --train.wandb.tags [owkin,c_cells]"
    done
done

for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_d_cells_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path d_cells.json \
                --train.data.metadata_path d_cells/outer=${OUTER}.parquet \
                --train.data.panel_path d_cells.yaml \
                --train.data.cache_dir protein/d_cells \
                --train.wandb.tags [owkin,d_cells]"
    done
done

for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2 3 4; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_g_cells_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path g_cells.json \
                --train.data.metadata_path g_cells/outer=${OUTER}.parquet \
                --train.data.panel_path g_cells.yaml \
                --train.data.cache_dir protein/g_cells \
                --train.wandb.tags [owkin,g_cells]"
    done
done
```

For `c_on_d`/`d_on_c`, `items_path`/`panel_path`/`cache_dir` stay fixed at `c_d_cells` (shared
item set and warmed cache); loop `--train.data.metadata_path` over the split's own outer folds —
`c_on_d` has 5 (0-4), `d_on_c` has 3 (0-2):

```bash
for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2 3 4; do
        uv run python scripts/train/supervised.py \
            --config configs/train/owkin/proteins/${CONFIG}.yaml \
            --debug=True \
            --train.data.items_path c_d_cells.json \
            --train.data.metadata_path c_on_d/outer=${OUTER}.parquet \
            --train.data.panel_path c_d_cells.yaml \
            --train.data.cache_dir protein/c_d_cells \
            --train.wandb.tags "[owkin, c_on_d]"
    done
done
```

Submit one dedicated sbatch loop per split direction, sweeping `CONFIG` and every outer fold that
direction has:

```bash
PARTITION=gpu-l40
for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2 3 4; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_c_on_d_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path c_d_cells.json \
                --train.data.metadata_path c_on_d/outer=${OUTER}.parquet \
                --train.data.panel_path c_d_cells.yaml \
                --train.data.cache_dir protein/c_d_cells \
                --train.wandb.tags [owkin,c_on_d]"
    done
done

for CONFIG in vision expr-token-vit expr-resmlp early-fusion; do
    for OUTER in 0 1 2; do
        sbatch \
            --account=rgottar1_spatial \
            --partition=$PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_d_on_c_${CONFIG}_outer${OUTER} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path c_d_cells.json \
                --train.data.metadata_path d_on_c/outer=${OUTER}.parquet \
                --train.data.panel_path c_d_cells.yaml \
                --train.data.cache_dir protein/c_d_cells \
                --train.wandb.tags [owkin,d_on_c]"
    done
done
```

To smoke-test a single run before committing to the full sweep, add `--debug true`
(shrinks batch size, worker count, and epoch/batch limits) and drop the cache override
to skip caching entirely:

```bash
uv run python scripts/train/supervised.py \
    --config configs/train/owkin/proteins/early-fusion.yaml \
    --train.data.items_path c_cells.json \
    --train.data.metadata_path c_cells/outer=0.parquet \
    --train.data.panel_path c_cells.yaml \
    --train.data.cache_dir null \
    --debug true
```
