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
#CH_C_518a_x2 CH_C_523a_x2
for SAMPLE_ID in CH_C_525a_x2 CH_C_526a_x1 CH_C_527a_x2 CH_C_527a_xr \
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
`data_dir`/`tile_px`/`stride_px` live in `configs/artifacts/owkin/cells.yaml`, so no CLI overrides
are needed:

```bash
uv run python scripts/artifacts/create_items.py --config configs/artifacts/owkin/cells.yaml
```

### Compute per-tile statistics for `all.json`

Computes `num_transcripts`/`num_unique_transcripts`/`num_cells`/`num_unique_cells` for every item in
`items/all.json` (needed before `filter_items`/`create_artifacts` can threshold on them). Always
targets the source `all.json`, independent of the config's `items.name` (which names the *filtered*
subset, e.g. `cells`):

```bash
uv run python scripts/artifacts/compute_items_stats.py --config configs/artifacts/owkin/cells.yaml
```

### Compute stats for a filtered subset

After `filter_items.py` (or `create_artifacts.py`) produces a filtered `items/<name>.json` (e.g.
`cells.json`, using the thresholds under `artifacts.items.filter` in `cells.yaml` against the
default stats above), point `compute_items_stats.py` at it directly via `--items-path` — resolved
relative to `items/`, or pass an absolute path:

```bash
uv run python scripts/artifacts/compute_items_stats.py \
    --config configs/artifacts/owkin/cells.yaml \
    --items-path cells.json
```

### Create filtered artifacts (c_cells / d_cells / g_cells)

Runs the full pipeline (filter `all.json` using the default stats -> split -> panel -> stats on
the filtered subset) for each per-organ-group config. `items/all.json` and `statistics/all.parquet`
must already exist (see above) — `create_artifacts.py` skips those two stages if so and goes
straight to filtering:

```bash
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/owkin/c_cells.yaml
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/owkin/d_cells.yaml
uv run python scripts/artifacts/create_artifacts.py --config configs/artifacts/owkin/g_cells.yaml
```

### Build per-item-set source panels

`panels/owkin/create_panels.py` intersects each sample's `feature_universe.txt` across an
artifacts config's `items.filter.include_ids` and writes the result to
`DATA_DIR/03_output/owkin/panels/<items.name>.yaml`. Run once after `create_artifacts.py`
for a new item-set variant (or after `--overwrite true` to rebuild):

```bash
uv run python panels/owkin/create_panels.py
```

### Warm the tile cache

`warmup_cache.py` populates `DATA_DIR/03_output/owkin/cache/protein/<ITEMS>` before
training so the first epoch doesn't pay tile-read cost. `ITEMS` selects which item
set/panel/split to warm — must match one of the artifacts configs above (`c_cells`,
`d_cells`, `c_d_cells`, `g_cells`):

```bash
for ITEMS in c_cells d_cells c_d_cells; do
    uv run python scripts/artifacts/warmup_cache.py \
        --config configs/train/owkin/proteins/early-fusion.yaml \
        --train.data.items_path ${ITEMS}.json \
        --train.data.metadata_path ${ITEMS}/outer=0-inner=0-seed=0.parquet \
        --train.data.panel_path ${ITEMS}.yaml \
        --train.data.cache_dir protein/${ITEMS}
done
```

### Train protein-prediction models

Each config under `configs/train/owkin/proteins/` (`early-fusion`, `expr-token-vit`,
`expr-resmlp`, `late-fusion-tile`, `vision`) fixes an architecture; override `--train.data.*` to pick
the item-set/panel/cache to train on. Keep `ITEMS` consistent with whichever cache you
warmed above:

```bash
for ITEMS in c_cells d_cells c_d_cells; do
    for CONFIG in early-fusion expr-token-vit expr-resmlp late-fusion-tile vision; do
        uv run python scripts/train/supervised.py \
            --config configs/train/owkin/proteins/${CONFIG}.yaml \
            --train.data.items_path ${ITEMS}.json \
            --train.data.metadata_path ${ITEMS}/outer=0-inner=0-seed=0.parquet \
            --train.data.panel_path ${ITEMS}.yaml \
            --train.data.cache_dir protein/${ITEMS} \
            --train.wandb.tags "[owkin, ${ITEMS}]"
    done
done
```

Submit the same sweep as one SLURM job per `(ITEMS, CONFIG)` pair:

```bash
for ITEMS in c_cells d_cells c_d_cells; do
    for CONFIG in early-fusion expr-token-vit expr-resmlp late-fusion-tile vision; do
        sbatch \
            --account=rgottar1_spatial \
            --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 \
            --output=$HOME/logs/%j.out \
            --job-name=owkin_proteins_${ITEMS}_${CONFIG} \
            --wrap="uv run python scripts/train/supervised.py \
                --config configs/train/owkin/proteins/${CONFIG}.yaml \
                --train.data.items_path ${ITEMS}.json \
                --train.data.metadata_path ${ITEMS}/outer=0-inner=0-seed=0.parquet \
                --train.data.panel_path ${ITEMS}.yaml \
                --train.data.cache_dir protein/${ITEMS} \
                --train.wandb.tags [owkin,${ITEMS}]"
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
    --train.data.metadata_path c_cells/outer=0-inner=0-seed=0.parquet \
    --train.data.panel_path c_cells.yaml \
    --train.data.cache_dir null \
    --debug true
```
