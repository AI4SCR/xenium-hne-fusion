# Xenium x H&E Fusion

Research code for ["Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer"](https://openreview.net/forum?id=h2GcySraTP) (ICLR 2026 Workshop LMRL) — an early-fusion multimodal transformer that integrates subcellular Xenium transcript readouts directly into the ViT token stream for gene expression prediction and downstream patient-level MIL tasks.

Primary results are on an internal Xenium cohort (BEAT); we also benchmark on [`hest1k`](https://arxiv.org/abs/2406.16192) using splits and panels from the [HESCAPE](https://arxiv.org/abs/2508.01490) benchmark.

## Pipeline overview

```
raw data
  └─► structure + process (scripts/data/)
        └─► items + splits + panels (scripts/artifacts/)
              └─► supervised training (scripts/train/supervised.py)
                    └─► prediction cache (scripts/artifacts/cache_predictions.py)
                          └─► MIL training (scripts/train/mil.py)
```

Each stage writes managed outputs under `DATA_DIR/03_output/<name>/`. All paths in configs are relative to that root.

## Project structure

```text
xenium-hne-fusion/
├── src/xenium_hne_fusion/   # reusable package code
├── scripts/data/            # dataset structuring and processing entrypoints
├── scripts/artifacts/       # items, splits, panels, stats, and MIL cache/metadata
├── scripts/train/           # training entrypoints (supervised and MIL)
├── scripts/eval/            # W&B score plots and paired tests
├── configs/data/            # dataset processing configs
├── configs/artifacts/       # artifact generation configs
├── configs/train/           # supervised training configs
├── configs/mil/             # MIL training configs
├── configs/eval/            # evaluation configs
├── slurm/                   # Slurm experiment command references
├── ray/                     # Ray submission helpers and command references
├── tests/                   # pytest suite
├── data/                    # managed raw / structured / processed / output data
└── results/                 # local outputs, not tracked dataset artifacts
```

## Quickstart

This repo depends on [`ai4bmr-learn`](https://github.com/AI4SCR/ai4bmr-learn), which is installed as a local editable dependency. Clone it first and keep it on the latest `main`:

```bash
git clone git@github.com:AI4SCR/ai4bmr-learn.git
cd ai4bmr-learn && git pull origin main && cd ..
```

Then clone and set up this repo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:AI4SCR/xenium-hne-fusion.git && cd xenium-hne-fusion
uv sync
cp .env.example .env   # then fill in the values
```

`uv sync` resolves `ai4bmr-learn` from the sibling directory via the editable path in `pyproject.toml`. If it can't be found, check that both repos are cloned at the same level.

Use `uv run ...` for all entrypoints — it loads `.env` automatically.

### `.env` variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for gated HEST-1k assets |
| `WANDB_API_KEY` | W&B key for training tracking and eval plots |
| `ACCOUNT` | Slurm account (cluster only) |
| `DATA_DIR` | Root for `00_raw/`, `01_structured/`, `02_processed/`, `03_output/` |
| `HEST1K_RAW_DIR` | Raw HEST-1k location |
| `BEAT_RAW_DIR` | Raw BEAT location (internal dataset) |

## Key scripts

| Script | What it does |
|--------|-------------|
| `scripts/data/structure_<dataset>.py` | Download + validate raw data, build `01_structured/` |
| `scripts/data/process_<dataset>.py` | Tile slides, extract per-tile artifacts, build `02_processed/` |
| `scripts/data/create_items.py` | Build `03_output/<name>/items/all.json` tile inventory |
| `scripts/artifacts/create_artifacts.py` | Filter items, create splits and gene panels |
| `scripts/train/supervised.py` | Train a supervised tile-level model (expression / cell types) |
| `scripts/artifacts/create_mil_metadata.py` | Join items with clinical labels for MIL |
| `scripts/artifacts/cache_predictions.py` | Run pretrained model inference; write per-patient bags (GPU) |
| `scripts/train/mil.py` | Train a MIL aggregator on top of cached embeddings |
| `scripts/eval/plot_wandb_scores.py` | Fetch W&B runs and produce score plots + paired tests |

## Datasets

### BEAT (internal)

Internal Xenium + H&E cohort. Data access is institutional. Preparation and training commands: [slurm/beat.md](slurm/beat.md).

### HEST-1k / HESCAPE

Public Xenium + H&E benchmark. Requires `HF_TOKEN`. HESCAPE uses fixed outer-fold splits and gene panels from the [HESCAPE benchmark](https://arxiv.org/abs/2508.01490). Commands: [slurm/hescape.md](slurm/hescape.md) and [slurm/hest1k.md](slurm/hest1k.md).

## Configs

Configs are YAML files loaded via `jsonargparse`. Individual fields can be overridden on the CLI:

```bash
uv run python scripts/train/supervised.py \
    --config configs/train/beat/cell_types/early-fusion.yaml \
    --debug true
```

Key training paths (`data.items_path`, `data.metadata_path`, `data.panel_path`, `data.cache_dir`) are relative to `DATA_DIR/03_output/<name>/` and resolved at runtime.

## Cluster runs

All exact submission commands live in the referenced `.md` files — use them as the authoritative command reference, not this README.

| File | Contents |
|------|----------|
| [slurm/beat.md](slurm/beat.md) | BEAT data prep, supervised training, evaluation |
| [slurm/hescape.md](slurm/hescape.md) | HESCAPE artifact creation and training sweep |
| [slurm/hest1k.md](slurm/hest1k.md) | HEST-1k data structuring and processing |
| [slurm/mil.md](slurm/mil.md) | MIL pipeline: chained cache → training jobs for all run IDs and aggregators |
| [ray/hescape.md](ray/hescape.md) | Ray equivalents for HESCAPE training |
| [ray/beat.md](ray/beat.md) | Ray equivalents for BEAT training |

## Task tracking

Current experiment status and next tasks: [tasks.md](tasks.md).

## Citation

```bibtex
@inproceedings{martinelli2026,
  title     = {Learning Joint Morpho-Molecular Tissue Representations with a Multimodal Transformer},
  author    = {Adriano Martinelli and Bernd Illing and Isinsu Katircioglu and Alice Driessen and Fei Tang and Robert Berke and Raphael Gottardo and Marianna Rapsomaniki},
  booktitle = {ICLR 2026 Workshop on Learning and Mining with Representation Learning (LMRL)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=h2GcySraTP},
}
```