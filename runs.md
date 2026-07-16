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
