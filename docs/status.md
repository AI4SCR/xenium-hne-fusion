# Project status: handover

Last updated: 2026-09-24. Written so another agent can pick up the work.

## 1. Where things are

| What | Where |
|---|---|
| PI project report (HTML artifact) | https://claude.ai/artifact/BioYsEd9nhHvTcuPnLbwAv (source: session scratchpad, republish via Artifact `url`) |
| Paper draft | `paper/` (owkin branch). ICLR 2026 LMRL workshop version: https://openreview.net/forum?id=h2GcySraTP |
| Current branch | `owkin` (Owkin protein work, 71 commits ahead of `main`; BEAT/HEST1k/MIL code removed here) |
| BEAT training worktree | `../xenium-hne-fusion-c1759b4` (detached at `c1759b4`, has its own `.venv` and `.env`) |
| Unused worktree | `../xenium-hne-fusion-main` (at `main` = `a5fd135`, cannot train BEAT, see §3; safe to `git worktree remove`) |

### Data roots (`/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/`)

| Root | Contents | Used by |
|---|---|---|
| `xenium-hne-fusion-v0/03_output/` | `beat`, `hest1k`, `owkin` (v0) | **All** BEAT cell-type / expression W&B runs since 2026-04-20, Owkin protein-v0 |
| `xenium-hne-fusion-v4/03_output/` | `owkin` only | Owkin protein-v1 |
| `xenium-hne-fusion/03_output/` | empty | nothing |
| `/raid/ray/shared/.../xenium-hne-fusion/` | Ray-cluster copy | 40 early cell-v0 runs (2026-04-13..17), no cache |

`.env` sets `DATA_DIR=.../xenium-hne-fusion-v0`.

## 2. Environment gotchas

- **`uv run` does NOT load `.env`** (CLAUDE.md claims it does). Prefix ad-hoc commands with
  `UV_ENV_FILE=.env uv run ...`. Scripts that call `load_dotenv()` work regardless.
- **W&B**: the valid key is only in the project `.env` (entity `chuv`). `~/.netrc` holds a revoked key that
  wandb falls back to without `.env`. Do **not** run `wandb login` or edit `~/.netrc`, `~/.bashrc`, `~/.env`.
- **Slurm**: GPU partition `gpu-rtx` (Quadro RTX 8000), QoS **`dcsr`** (not "dscr"). An unknown QoS silently
  falls back to `gpu-normal`. Logs go to `$HOME/logs/%j.out`.

## 3. Why BEAT jobs must run from commit `c1759b4`

- BEAT caches `v0/03_output/beat/cache/{cell_types/default, expression/{default,hvg-50,hvg-100}/cells/outer=N-inner=0-seed=0}`
  were written in Apr 2026. They store **raw, untransformed** targets (integer counts) under the key `target`.
- `main` HEAD `a5fd135` ("owkin", 2026-07-10) changed `datasets/tiles.py`. It now stores and reads the target under
  the task name (`item['cell_types']`, `item['expression']`) and changed the default `cell_type_col`. The result is
  `KeyError: 'cell_types'` on the existing caches.
- No cache in the new key format exists. No W&B run has ever used one.
- Parent `c1759b4` reads `item['target']` and applies `target_transform` (log1p) **after** loading, so there is no double
  transform. It is also the code the Jun–Jul baselines ran on, so it gives a controlled comparison. We chose it over
  re-caching (318k items, ~143 GB).
- `c1759b4`'s `supervised.py` whitelists `wandb.name` (`early-fusion`, `early-fusion-vit`, `late-fusion-{tile,token,token-vit}`,
  `expr-{token,tile,token-vit}`, `vision`). Mark variants with `--wandb.group` and tags, not new names.
- **Fixing HEAD** (open): either migrate the loader to read the legacy `target` key, or re-cache with a script that caches
  raw targets only and asserts no transform has been applied before saving.

## 4. Running experiment: early fusion with linear expression ingestion

**Question.** Does the image add anything over a strong expression model?

**Background.** Expression-only with a ViT-S transcript encoder beat the paper's early fusion. That early fusion
squeezes each patch's 380 genes through a 32-d, 2-layer ReLU MLP, then `Linear(32→384)`. Expression-only ViT-S instead
uses a full-rank `Linear(380→384)` patch embedding. Both run the transcripts through the same ViT-S blocks. The
comparison was therefore confounded by the 32-d bottleneck.

**Change.** Early fusion with `expr_encoder_kws={output_dim: 384, num_hidden_layers: 0}` (a plain `Linear(G→384)`)
and `use_proj=false`. Everything else matches the expr-only ViT-S runs (lr 1e-4 head / 1e-5 backbone, 35 epochs, bs 256,
accumulate 2, `max_time` 6 h). Backbone params: 21.8 M (ViT-S 21.67 M + 146 k linear), which confirms the override took effect.

W&B: `--wandb.name early-fusion --wandb.group early-fusion-linear`, tags `[beat, lung, linear-expr]`.

| Task | Fold | Slurm job | W&B project / run |
|---|---|---|---|
| cell types | 0 | 65122809 | `xe-hne-fus-cell-v1/ot1pgfni` (slower node, ~5 h) |
| cell types | 1 | 65122830 | `xe-hne-fus-cell-v1/jpjepvt8` |
| cell types | 2 | 65122831 | `xe-hne-fus-cell-v1/tlo50d9c` |
| cell types | 3 | 65122832 | `xe-hne-fus-cell-v1/l8vs7ols` |
| HVG-100 genes | 0 | 65122833 | `xe-hne-fus-expr-v0/yougc371` |
| HVG-100 genes | 1 | 65122834 | `xe-hne-fus-expr-v0/c44l6rqk` |
| HVG-100 genes | 2 | 65122835 | `xe-hne-fus-expr-v0/42sp5wbq` |
| HVG-100 genes | 3 | 65122836 | `xe-hne-fus-expr-v0/fhd8ooec` |
| HVG-50 genes | 0 | 65123169 | `xe-hne-fus-expr-v0/n0c3o6fr` |
| HVG-50 genes | 1 | 65123170 | `xe-hne-fus-expr-v0/jihqpk5k` |
| HVG-50 genes | 2 | 65123171 | `xe-hne-fus-expr-v0/otx0gemw` |
| HVG-50 genes | 3 | 65123172 | `xe-hne-fus-expr-v0/dt030tk6` |

Submitted 2026-09-24 ~15:20 (HVG-50 ~16:45). Expected runtime is 3–5 h.

### Submission command (reuse for new variants)

```bash
cd ../xenium-hne-fusion-c1759b4
WT=$PWD
OVR="--backbone.expr_encoder_kws '{output_dim: 384, num_hidden_layers: 0}' --backbone.use_proj false \
     --trainer.max_time 00:06:00:00 --wandb.name early-fusion --wandb.group early-fusion-linear \
     --wandb.tags '[beat, lung, linear-expr]'"
TASK=expression   # or cell_types
for OUTER in 0 1 2 3; do
  SPLIT_NAME="outer=${OUTER}-inner=0-seed=0"
  PANEL_PATH="hvg-100/cells/${SPLIT_NAME}.yaml"   # cell_types: default.yaml ; hvg-50: hvg-50/cells/...
  PANEL_NAME="${PANEL_PATH%.yaml}"
  sbatch --chdir=$WT --partition=gpu-rtx --qos=dcsr --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=06:30:00 \
    --output=$HOME/logs/%j.out --job-name=${TASK}-early-linear-${OUTER} \
    --wrap="UV_ENV_FILE=.env uv run python scripts/train/supervised.py \
      --config configs/train/beat/${TASK}/early-fusion.yaml \
      --data.items_path cells.json --data.metadata_path cells/${SPLIT_NAME}.parquet \
      --data.panel_path ${PANEL_PATH} --data.cache_dir=${TASK}/${PANEL_NAME} ${OVR}"
done
```

### Baselines (same split files `cells/outer=N-inner=0-seed=0.parquet`, one run per fold)

Final **val** Pearson (test in parentheses):

| Task | Model | fold 0 | fold 1 | fold 2 | fold 3 | mean |
|---|---|---|---|---|---|---|
| cell types | early fusion MLP-32 (add, unfrozen) | 0.744 (0.736) | 0.741 (0.733) | 0.755 (0.739) | 0.744 (0.732) | 0.746 (0.735) |
| cell types | expr-only ViT-S (`expr-token`, `vit_small`) | 0.765 (0.799) | 0.775 (0.752) | 0.786 (0.797) | 0.791 (0.768) | 0.779 (0.779) |
| HVG-100 | early fusion MLP-32 (add, unfrozen) | 0.779 (0.785) | 0.810 (0.788) | 0.794 (0.755) | 0.801 (0.809) | 0.796 (0.784) |
| HVG-100 | expr-only ViT-S | 0.805 (0.816) | 0.819 (0.808) | 0.809 (0.783) | 0.832 (0.845) | 0.816 (0.813) |
| HVG-50 | early fusion MLP-32 (add, unfrozen) | 0.804 | 0.809 | 0.806 | 0.802 | 0.805 |
| HVG-50 | expr-only ViT-S | **not run** | | | | |

Other refs: cell-types expr-only **ViT-B** (fold 0 only, `mlbrojzj`) test 0.852. Late-fusion-token-vit (fold 0,
`kj9m62uc`/`z4ca29zp`) test 0.821/0.823.

**Mid-training snapshot (~70 min, epoch 9–14 of 35), val Pearson:** cell types 0.774 / 0.789 / 0.805 / 0.800;
HVG-100 0.817 / 0.841 / 0.830 / 0.830. At that point it was already at or above expr-only ViT-S on 7 of 8 folds.
This is not a result: it is the latest step, not the best-checkpoint test score.

### Baseline run selection (why these runs)

We ran many BEAT early-fusion experiments. For each (task, model) the baseline is exactly **one finished run per
fold** that matches the new runs on everything except the expression ingestion (MLP-32 vs linear) or the image
(expr-only). Selection was done over a full W&B pull of `xe-hne-fus-{cell-v0,cell-v1,expr-v0}` (full configs).

Rules for a run to be kept:
1. **Same data**: root `xenium-hne-fusion-v0`, items `cells.json`, splits `cells/outer=N-inner=0-seed=0.parquet`.
2. **Same targets**: cell types with `default.yaml`. HVG-100 or HVG-50 with the per-fold panels
   `hvg-{100,50}/cells/outer=N-inner=0-seed=0.yaml`, identified by `len(target_panel)` = 100 / 50.
3. **Same model family**: ViT-S image encoder **unfrozen**, `fusion_strategy=add` (the new runs use add), expression
   encoder `mlp` (dim 32) for MLP-32. For expr-only, `wandb.name=expr-token` with `expr_encoder_name=vit_small_patch16_224`.
4. `state=finished` with test metrics. Prefer the project that holds the expr-only ViT-S runs (`cell-v1` for cell types).

| Group in W&B (not used) | Runs | Why excluded |
|---|---|---|
| `cell-v0`, root `/raid/ray/...` (Apr 13–17) | 12 early, 4 expr-token | Ray-cluster copy of the data, no cache, older code; superseded by the v0-root reruns |
| `cell-v0`, v0 root, early add unfrozen (Apr 20–May 5) | 10 (4 finished) | Duplicates and crashes; `cell-v1` (May 7) is a clean 1-run-per-fold rerun in the same project as expr-only ViT-S |
| `cell-v1` `ryydvint` (Jul 1, fold 0) | 1 | `items_with_conch_labels.json` (different item set) and no test metrics |
| Frozen-image (`freeze_morph_encoder=True`) | 4 per task | Different variable (frozen ViT); the new runs fine-tune the ViT |
| `concat` fusion | 4 per task | New runs use `add` (concat doubles the sequence length) |
| `expr-v0` `default.yaml` (380 core → 100 add-on genes, Apr 17–26) | many | The paper's gene task, not HVG. No expr-only ViT-S exists there |
| `expr-v0` `expr.json` + `expr-hvg-outer=N.yaml` (Apr 18–19) | 12 early, 4 expr | Older HVG variant on a different item set (`expr.json`). The Jun expr-only ViT-S runs used `cells.json` + `hvg-100/cells/...` |
| `expr-token` MLP-32 (`ee=mlp`) | 4 per task | Weak expr-only baseline (kept in the report, not in this comparison) |
| `expr-token` ViT-B (`mlbrojzj`) | 1 (fold 0) | Only one fold; different capacity |

Selected run IDs (fold 0..3) are hard-coded in `scripts/eval/compare_linear_expr.py`:

| Task | early MLP-32 | expr-only ViT-S | early linear (new) |
|---|---|---|---|
| cell types (`cell-v1`) | 40utmvmw, j5x4suw2, owzcohia, fif5wuld | w0y2mwg5, 005s9g5a, pbi7jz39, u2l1eisi | ot1pgfni, jpjepvt8, tlo50d9c, l8vs7ols |
| HVG-100 (`expr-v0`) | uaepyp93, hdkim0ma, z2shyn16, 5a6crdzl | 3nh0f2lw, fk4lzu7l, buzc1v5e, juonbqiu | yougc371, c44l6rqk, 42sp5wbq, fhd8ooec |
| HVG-50 (`expr-v0`) | gl0w9chl, 6361tn0m, bngpxrkn, 4nwbprmb | not run | n0c3o6fr, jihqpk5k, otx0gemw, dt030tk6 |

Caveats:
- Expr-only ViT-S runs stopped early (epochs 8–20). The MLP-32 HVG runs ran all 35 epochs. Metrics are for the best
  checkpoint (by val), so this is fine, but do not compare "last epoch" values.
- The MLP-32 baselines ran on older commits (Apr–May) than the expr-only ViT-S runs (Jun) and the new runs (`c1759b4`).
  The loader semantics are the same (same caches, `target` key), but this is not bit-identical code.

### When the runs finish

1. Run `UV_ENV_FILE=.env uv run python scripts/eval/compare_linear_expr.py` (needs network). It prints per-fold
   test/val Pearson and paired deltas `early_linear − {early_mlp32, expr_vit_s}`, and writes `compare_linear_expr.csv` to the cwd.
   Add the HVG-50 expr-only ViT-S run IDs to `BASELINES` once they exist.
2. Update the report artifact (sections "A stronger expression-only baseline", "Open questions", "Next steps").
   Replace the wording "early fusion with the ViT transcript encoder" with the bottleneck confound described here.

## 5. Open items / next steps

- [ ] **HVG-50 expr-only ViT-S baseline** (4 folds), still missing. Use `configs/train/beat/expression/expr-token-vit.yaml`
      (its `wandb.name` is `expr-token`) from the `c1759b4` worktree, with panel `hvg-50/cells/...`, `max_time` 6 h. Not submitted; awaiting user OK.
- [ ] Evaluate the 12 runs above and update the report.
- [ ] Optional stronger check: early-linear with ViT-B on cell types vs expr-only ViT-B (only 1 fold exists).
- [ ] Fix `main` HEAD's loader or re-cache (§3). Fix the CLAUDE.md `.env` claim (§2).
- [ ] Remove `../xenium-hne-fusion-main` worktree if unused.

## 6. Key findings so far (details in the report)

- BEAT paper: fusion beats unimodal *MLP* baselines (cell types 0.761 vs 0.554 morph / 0.655 expr). The gene table
  reproduces from `xe-hne-fus-expr-v0` `default.yaml` runs within ±0.01. The cell-table ViT-B / MLP-128 / core / add-on rows
  cannot be traced in this repo's W&B projects (the older `xe-hne` project is dated 2025-12-25 with different keys).
- Paper capacity paragraph compares different panels (0.688 is add-on ViT-S). With the panel fixed: 0.742 → 0.761.
- Best gene model in the paper table is late-concat token with frozen ViT-S (Spearman 0.820), not early fusion.
- HESCAPE: late fusion beats early fusion on all 5 panels. HEST-1k panels are tiny (42–78 source → 10–20 target genes).
- Inference ablations (BEAT cell types): within-tile expr shuffle −0.008, cross-tile expr swap −0.50, cross-tile image
  swap −0.04, zero image −0.47.
- Owkin protein (v1, `xenium-hne-fusion-v4`): expression-only models win every setting. Cross-indication C↔D collapses.
  Within-sample, vision reaches 0.69–0.80, which indicates a strong sample shift. Owkin v0 splits leak (x2/xr sections of the same
  specimen across fit/test in folds 0 and 3); do not quote v0.
- MIL (`xe-hne-fus-mil-v0`, binary `metadata.7`): chance level (AUC ≤ 0.64) for every encoder.
- Batch correction (Owkin c_cells): metrics disagree on a winner, and no bio-conservation metric has been computed yet.
