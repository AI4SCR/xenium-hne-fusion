Code for the paper [Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer](https://openreview.net/forum?id=h2GcySraTP).

# Xenium x HnE Fusion

Research codebase for fusing Xenium spatial transcriptomics with H&E whole-slide images.
Supports two dataset families:

- `hest1k`: HEST-1k human Xenium data downloaded from Hugging Face
- `beat`: internal BEAT data arranged under a local raw root

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd xenium-hne-fusion
uv sync
cp .env.example .env
```

`.env` holds machine-specific paths and tokens:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=

DATA_DIR=data
HEST1K_RAW_DIR=
BEAT_RAW_DIR=
```

`uv run ...` loads `.env` automatically. For notebooks or ad hoc sessions:

```python
from dotenv import load_dotenv

load_dotenv()
```

## Repo structure

```text
xenium-hne-fusion/
├── src/xenium_hne_fusion/      # importable package
├── scripts/data/               # data pipeline entrypoints
├── scripts/train/              # training entrypoints
├── configs/data/               # processing configs
├── configs/train/              # training configs
├── panels/                     # checked-in panel artifacts
├── slurm/                      # Slurm wrappers
├── ray/                        # Ray submission helpers
├── tests/
├── data/                       # managed raw/structured/processed/output data
└── results/                    # local experiment outputs
```

## Managed data layout

Managed paths are derived from `DATA_DIR` and the dataset `name`.
For `name: hest1k`:

```text
DATA_DIR/01_structured/hest1k/
DATA_DIR/02_processed/hest1k/
DATA_DIR/03_output/hest1k/
```

Typical layout:

```text
data/
├── 00_raw/
│   ├── hest1k/
│   │   ├── HEST_v1_3_0.csv
│   │   ├── wsis/
│   │   └── transcripts/
│   └── beat/
│       ├── metadata.parquet
│       └── <sample_id>/
│           ├── region.tiff
│           └── transcripts/transcripts.parquet
│
├── 01_structured/<name>/
│   ├── metadata.csv|metadata.parquet
│   └── <sample_id>/
│       ├── wsi.tiff
│       ├── transcripts.parquet
│       ├── wsi.png
│       ├── transcripts.png
│       ├── tissues.parquet
│       └── tiles/{tile_px}_{stride_px}.parquet
│
├── 02_processed/<name>/
│   ├── metadata.parquet
│   └── <sample_id>/
│       ├── feature_universe.txt
│       └── {tile_px}_{stride_px}/
│           └── <tile_id>/
│               ├── tile.pt
│               ├── transcripts.parquet
│               ├── expr-kernel_size=16.parquet
│               ├── tile.png
│               ├── transcripts.png
│               └── transcripts_top5_feats.png
│
└── 03_output/<name>/
    ├── items/
    │   ├── all.json
    │   └── <items_name>.json
    ├── statistics/
    │   └── all.parquet
    ├── figures/
    │   └── items/
    │       ├── stats/
    │       │   └── <items_name>/
    │       │       └── *.png
    │       └── gene_panel_overlap/
    │           ├── <items_name>.pdf
    │           └── <items_name>.png
    ├── splits/
    │   └── <split_name>/
    │       └── outer=0[-inner=0]-seed=<seed>.parquet
    ├── panels/
    │   └── <panel_name>.yaml
    ├── cache/
    └── logs/
```

Key outputs:

- `items/all.json`: complete tile set used for source-item stats
- `items/<items_name>.json`: filtered subset
- `figures/items/stats/<items_name>/`: tile-stat plots
- `figures/items/gene_panel_overlap/<items_name>.{pdf,png}`: sample gene-panel overlap heatmaps
- `splits/<split_name>/`: split parquet files
- `panels/`: generated panel YAMLs

## Processing config schema

The data pipeline uses one nested config schema for both datasets:

```yaml
name: hest1k
tiles:
  tile_px: 512
  stride_px: 256
  mpp: 0.5
  kernel_size: 16
  predicate: within
filter:
  species: Homo sapiens
  organ: null
  disease_type: null
  include_ids: null
  exclude_ids: null
items:
  name: default
  filter:
    organs: null
    num_transcripts: 100
    num_unique_transcripts: null
    num_cells: null
    num_unique_cells: null
split:
  name: default
  test_size: 0.25
  val_size: 0.25
  stratify: false
  target_column_name: null
  encode_targets: false
  nan_value: -1
  use_filtered_targets_for_train: false
  include_targets: null
  group_column_name: null
  random_state: 0
```

Examples:

- [configs/artifacts/hest1k/expr.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/artifacts/hest1k/expr.yaml)
- [configs/artifacts/hest1k/breast.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/artifacts/hest1k/breast.yaml)
- [configs/artifacts/hest1k/lung.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/artifacts/hest1k/lung.yaml)
- [configs/artifacts/hest1k/pancreas.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/artifacts/hest1k/pancreas.yaml)
- [configs/artifacts/beat/unil/expr.yaml](/Users/adrianomartinelli/projects/xenium-hne-fusion/configs/artifacts/beat/unil/expr.yaml)

`scripts/data/run_hest1k.py`, `scripts/data/run_beat.py`, `scripts/artifacts/filter_items.py`, and `scripts/artifacts/create_splits.py` all use this schema. `filter.include_ids` and `filter.exclude_ids` are mutually exclusive:

```bash
--filter.include_ids '[TENX116]'
--items.name breast
--items.filter.organs '[Breast]'
--split.name breast
--split.group_column_name sample_id
```

## Local CLI

### Manual stages

#### HEST1K structure

```bash
uv run scripts/data/structure_hest1k.py hest1k \
  --config_path configs/data/local/hest1k.yaml
```

#### BEAT structure

```bash
uv run scripts/data/structure_beat.py beat \
  --config_path configs/data/local/beat.yaml
```

#### Metadata

```bash
uv run python scripts/data/process_metadata.py \
  --config configs/data/local/hest1k.yaml
```

#### HEST1K processing

```bash
uv run python scripts/data/process_hest1k.py \
  --config configs/data/local/hest1k.yaml
```

#### Items

```bash
uv run python scripts/data/create_items.py \
  --config configs/data/local/hest1k.yaml
```

#### Source item stats

Intended to compute summary stats for `items/<items.name>.json` using the artifacts config.
`scripts/artifacts/compute_items_stats.py` parses `ArtifactsConfig`, so it expects an artifacts config.

```bash
uv run scripts/artifacts/compute_items_stats.py \
  --config configs/artifacts/hest1k/expr.yaml
```

#### Filtered items

```bash
uv run scripts/artifacts/filter_items.py \
  --config configs/artifacts/hest1k/breast.yaml
```

#### Splits

```bash
uv run scripts/artifacts/create_splits.py \
  --config configs/artifacts/hest1k/breast.yaml
```

#### Panel

```bash
uv run scripts/artifacts/create_panel.py \
  --config configs/artifacts/hest1k/breast.yaml
```

#### Gene-panel overlap diagnostics

Use this to inspect sample-level `feature_universe.txt` overlap for the current filtered item set.
The plot title reports the minimum and maximum pairwise intersection sizes and the size of the
intersection shared across all selected samples.

```bash
uv run python scripts/artifacts/report_feature_overlap.py \
  --config configs/artifacts/hest1k/breast.yaml
```

Outputs are written under `DATA_DIR/03_output/<name>/figures/items/gene_panel_overlap/`:

- `<items_name>.pdf`
- `<items_name>.png` at 300 DPI

#### All artifacts

`scripts/artifacts/create_artifacts.py` runs the full artifacts workflow for a single
artifacts config:

1. filter `items/all.json` into `items/<items.name>.json`
2. build `splits/<split.name>/`
3. create or validate `panels/<panel.name>.yaml` when `panel:` is set
4. compute `statistics/all.parquet` for the filtered items

```bash
uv run python scripts/artifacts/create_artifacts.py \
  --config configs/artifacts/hest1k/breast.yaml
```

If the config has no `panel:` section, panel creation is skipped. If `panel:` points to a
predefined panel, the script only checks that the target YAML already exists.

### End-to-end runners

The dataset runners execute the full pipeline:

1. structure raw inputs
2. detect tissues
3. tile and process samples
4. clean sample metadata
5. build `items/all.json`
6. compute `statistics/all.parquet`
7. write the filtered item set
8. write the split collection

#### HEST1K

```bash
uv run scripts/data/run_hest1k.py \
  --config configs/data/remote/hest1k.yaml \
  --name hest1k \
  --executor serial
```

#### BEAT

```bash
uv run scripts/data/run_beat.py \
  --config configs/data/remote/beat.yaml \
  --name beat \
  --executor serial
```

Both runners also support `--executor ray` and nested overrides. For example, a single-sample HEST1K run:

```bash
uv run scripts/data/run_hest1k.py \
  --config configs/data/local/hest1k.yaml \
  --name hest1k \
  --executor serial \
  --filter.include_ids '[TENX116]'
```

## Panel recipes and training path resolution

Panels are runtime YAML files consumed by training:
- `DATA_DIR/03_output/<name>/panels/*.yaml`

Panel creation is defined in the artifacts config. Keep the output filename in `data.panel_path`, and keep the recipe in a dedicated top-level `panel:` section:

```yaml
data:
  name: hest1k
  items_path: breast.json
  metadata_path: breast/outer=0-inner=0-seed=0.parquet
  panel_path: breast-hvg-outer=0-inner=0-seed=0.yaml

panel:
  name: breast-hvg-outer=0-inner=0-seed=0
  n_top_genes: 16
  flavor: seurat_v3
```

`scripts/artifacts/create_panel.py` accepts an artifacts config. It resolves the current artifacts layout, uses the fit split referenced by `split.name` and `panel.metadata_path`, and writes the panel YAML under `data.panel_path`.

Training configs under `configs/train/` resolve relative data paths as follows:

- `data.items_path` -> `DATA_DIR/03_output/<name>/items/`
- `data.metadata_path` -> `DATA_DIR/03_output/<name>/splits/`
- `data.panel_path` -> `DATA_DIR/03_output/<name>/panels/`
- `data.cache_dir` -> disabled when unset; relative paths resolve under `DATA_DIR/03_output/<name>/cache/`

If a training config uses absolute paths, they are left unchanged.

Examples:

- BEAT configs live under `configs/train/beat/...`
- HEST1K expression configs are organ-specific and live under `configs/train/hest1k/expression/<organ>/...`

Create panels:

```bash
uv run python scripts/artifacts/create_panel.py \
  --config configs/train/hest1k/expression/breast/early-fusion.yaml \
  --overwrite true

uv run python scripts/artifacts/create_panel.py \
  --config configs/train/beat/expression/early-fusion.yaml \
  --overwrite true
```

HEST1K expression configs already define a `panel:` recipe. For BEAT, the command is the same, but the chosen training config must also define a `panel:` recipe if you want `create_panel.py` to generate a panel instead of only checking that `data.panel_path` already exists.

Train a model:

```bash
uv run scripts/train/supervised.py \
  --config configs/train/beat/expression/early-fusion.yaml
```

The panel YAML must contain:

```yaml
source_panel:
  - GENE_A
target_panel:
  - GENE_B
```

## Slurm

Slurm wrappers live in `slurm/`:

```bash
slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k.yaml
slurm/run_beat_slurm.sh --config configs/data/remote/beat.yaml
slurm/train_hest1k_early_fusion_organ.sh breast
```

The dataset wrappers submit:

- one `samples` job per selected sample
- one `finalize` job that depends on all sample jobs

The wrapped jobs call the existing dataset pipelines in `scripts/data/run_hest1k.py` and `scripts/data/run_beat.py`, so the Slurm layer does not reimplement structuring, tissue detection, tiling, processing, metadata, items, stats, or splits.

Default resources:

- `8` CPUs
- `64G` RAM
- `08:00:00` wall time
- logs in `$HOME/logs`

Pass `--overwrite` to reprocess samples and overwrite finalized outputs.

```bash
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml
./slurm/run_hest1k_slurm.sh --config configs/data/remote/hest1k-breast.yaml --overwrite
./slurm/train_hest1k_early_fusion_organ.sh breast
```

The per-sample jobs run:

```bash
uv run python scripts/data/run_<dataset>.py \
  --config CONFIG_PATH \
  --executor serial \
  --stage samples \
  --filter.include_ids "[SAMPLE_ID]" \
  --filter.exclude_ids null
```

The finalization job runs once:

```bash
uv run python scripts/data/run_<dataset>.py \
  --config CONFIG_PATH \
  --executor serial \
  --stage finalize
```

See [slurm-cmds.md](slurm-cmds.md) for the full experiment command reference.

## Kaiko Ray

Ray helpers live under `ray/`:

```bash
bash ray/submit.sh "python scripts/train/supervised.py --config configs/train/beat/expression/early-fusion.yaml"
```

Pass `--entrypoint-num-gpus N` to reserve GPUs for the Ray entrypoint itself.

Useful helpers:

```bash
bash ray/submit.sh
bash ray/scripts/disk_space.sh
bash ray/submit.sh "bash ray/scripts/test_env.sh"
```

See [ray-cmds.md](ray-cmds.md) for the full experiment command reference.

## HESCAPE Runs

HESCAPE uses fixed (non-cross-validated) train/val/test splits drawn from HEST1K samples.
There is one split per organ group; no outer-fold loop is needed.

Organ groups and their training config counterparts:

| HESCAPE panel | Split path | Training config dir |
|---|---|---|
| breast | `hescape/breast/hescape.parquet` | `configs/train/hescape/expression/breast/` |
| bowel | `hescape/bowel/hescape.parquet` | `configs/train/hescape/expression/bowel/` |
| lung-healthy | `hescape/lung-healthy/hescape.parquet` | `configs/train/hescape/expression/lung-healthy/` |
| human-immuno-oncology | `hescape/human-immuno-oncology/hescape.parquet` | `configs/train/hescape/expression/human-immuno-oncology/` |
| human-multi-tissue | `hescape/human-multi-tissue/hescape.parquet` | `configs/train/hescape/expression/human-multi-tissue/` |

Splits: `DATA_DIR/03_output/hest1k/splits/hescape/<name>/hescape.parquet`.
Panels: pre-defined in `panels/hescape/<name>.yaml`, copied to `DATA_DIR/03_output/hest1k/panels/hescape/<name>.yaml`.

### 1. Create splits

Covers all HESCAPE organ groups in one command:

```bash
uv run python scripts/artifacts/create_hescape_splits.py
```

### 2. Validate panels

Based on instructions from the HESCAPE authors we ran the HVG analysis and determined the target genes ourselves.

```bash
uv run python scripts/artifacts/validate_hescape_panels.py
```

## Development

```bash
uv run pytest
uv add <pkg>
uv add --dev <pkg>
```

## Evaluation Plots

```bash
uv run python scripts/eval/plot_wandb_scores.py --config configs/artifacts/beat/kaiko/expr.yaml --project xe-hne-fus-expr-v0 --target expression --refresh true
uv run python scripts/eval/plot_wandb_scores.py --config configs/artifacts/beat/kaiko/expr-hvg.yaml --project xe-hne-fus-expr-v0 --target expression --refresh true
uv run python scripts/eval/plot_wandb_scores.py --config configs/artifacts/hest1k/breast.yaml --project xe-hne-fus-expr-v0 --target expression --refresh true
uv run python scripts/eval/plot_wandb_scores.py --config configs/artifacts/hescape/breast.yaml --project xe-hne-fus-expr-v0 --target expression --refresh true
```

Use `--wandb-filters` to pass a MongoDB-style filter dict to the W&B API.
Matching run IDs are intersected with the project cache before artifact selection.
For HEST1K, filter by organ and dataset tag simultaneously:

```bash
for organ in breast lung pancreas bowel; do
  uv run python scripts/eval/plot_wandb_scores.py \
    --config configs/artifacts/hest1k/${organ}.yaml \
    --project xe-hne-fus-expr-v0 \
    --target expression \
    --wandb-filters "{\"$and\": [{\"tags\": \"hest1k\"}, {\"tags\": \"${organ}\"}]}"
done
```

## Tasks

BEAT:

- [x] task: cell_types fusion_strategy: add panel: default learnable_gate: false
- [x] task: cell_types fusion_strategy: concat panel: default learnable_gate: false
- [x] task: cell_types fusion_strategy: add panel: default learnable_gate: true
- [ ] task: cell_types fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: cell_types fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: cell_types fusion_strategy: add panel: hvg learnable_gate: true
- [ ] task: expression fusion_strategy: add panel: default learnable_gate: false
- [ ] task: expression fusion_strategy: concat panel: default learnable_gate: false
- [ ] task: expression fusion_strategy: add panel: default learnable_gate: true
- [ ] task: expression fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: expression fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: expression fusion_strategy: add panel: hvg learnable_gate: true

HEST1k:

- [ ] task: expression organ: breast fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: expression organ: breast fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: expression organ: breast fusion_strategy: add panel: hvg learnable_gate: true
- [ ] task: expression organ: lung fusion_strategy: add panel: default learnable_gate: false
- [ ] task: expression organ: lung fusion_strategy: concat panel: default learnable_gate: false
- [ ] task: expression organ: lung fusion_strategy: add panel: default learnable_gate: true
- [ ] task: expression organ: lung fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: expression organ: lung fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: expression organ: lung fusion_strategy: add panel: hvg learnable_gate: true
- [ ] task: expression organ: pancreas fusion_strategy: add panel: default learnable_gate: false
- [ ] task: expression organ: pancreas fusion_strategy: concat panel: default learnable_gate: false
- [ ] task: expression organ: pancreas fusion_strategy: add panel: default learnable_gate: true
- [ ] task: expression organ: pancreas fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: expression organ: pancreas fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: expression organ: pancreas fusion_strategy: add panel: hvg learnable_gate: true
- [ ] task: expression organ: bowel fusion_strategy: add panel: default learnable_gate: false
- [ ] task: expression organ: bowel fusion_strategy: concat panel: default learnable_gate: false
- [ ] task: expression organ: bowel fusion_strategy: add panel: default learnable_gate: true
- [ ] task: expression organ: bowel fusion_strategy: add panel: hvg learnable_gate: false
- [ ] task: expression organ: bowel fusion_strategy: concat panel: hvg learnable_gate: false
- [ ] task: expression organ: bowel fusion_strategy: add panel: hvg learnable_gate: true
