# xenium-hne-fusion

Research codebase for Xenium spatial transcriptomics and H&E image fusion.

## Project structure

```
xenium-hne-fusion/
├── src/xenium_hne_fusion/   # Importable package (core logic, reusable modules)
├── scripts/                 # One-off analysis and experiment scripts
├── tests/                   # pytest tests
├── data/                    # Raw and processed inputs (not tracked, .gitkeep only)
├── results/                 # Outputs and model artifacts (not tracked)
├── figures/                 # Generated figures (not tracked)
├── pyproject.toml           # uv-managed project config
├── uv.lock                  # Locked dependency graph (committed)
└── .env                     # Machine-specific config (not tracked)
```

## Terminology

- **Tile**: rectangular WSI region cropped at a fixed mpp (e.g. 0.5 µm/px). The unit of data.
- **Patch**: a single ViT token — a sub-region of a tile (e.g. 16 × 16 px at the tile's resolution).
- Pipeline: WSI → **tile** (`extract_tiles`) → transform + **patch** tokens (`process_tiles`)

## Environment

Managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync                  # install all deps incl. dev
uv add <pkg>             # add a runtime dependency
uv add --dev <pkg>       # add a dev dependency
uv run pytest            # run tests
uv run python scripts/my_script.py
```

## Running scripts from a git worktree

When working in a git worktree (e.g. `.claude/worktrees/<branch>/`), `uv run` will fail
if `pyproject.toml` has editable dependencies with relative paths that don't resolve from
the worktree location (e.g. `ai4bmr-learn @ ../ai4bmr-learn`).

**Fix:** use the main project's venv with the worktree's `src/` on `PYTHONPATH`:

```bash
# from inside the worktree directory
PYTHONPATH=<worktree>/src <main_project>/.venv/bin/python scripts/data/my_script.py
```

Symlink `.env` and `data/` from the main project into the worktree:

```bash
ln -s <main_project>/.env <worktree>/.env
ln -s <main_project>/data <worktree>/data
```

The Bash tool runs with a minimal `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`). Homebrew tools
(e.g. `gh`, `jq`) are not on this path — call them by their full path:

```bash
/opt/homebrew/bin/gh pr create ...
```

This pattern applies to any project that uses editable installs with relative paths —
the installed package in the shared venv will point to the main branch; `PYTHONPATH`
overrides it with the worktree's modified source.

## Configuration

Machine-specific paths live in `.env` (not tracked). `uv run` loads it automatically;
for interactive sessions add `from dotenv import load_dotenv; load_dotenv()` at the top.

Required `.env` variables:

```bash
HF_TOKEN=hf_...             # HuggingFace token for gated datasets
DATA_DIR=data               # shared root for structured/processed/output data

# Dataset-specific raw roots
HEST1K_RAW_DIR=data/00_raw/hest1k

# BEAT (internal dataset — machine-specific paths)
BEAT_RAW_DIR=
```

Dataset-invariant settings (dataset `name`, tile size, sample filter) live in `configs/data/<dataset>.yaml`.
Model architecture and training hyperparams live in `configs/train/<variant>.yaml`.
Never hardcode paths in versioned YAMLs.

Managed paths are always derived as:

```bash
DATA_DIR/01_structured/<name>/
DATA_DIR/02_processed/<name>/
DATA_DIR/03_output/<name>/
```

Metadata is split across two levels:

```bash
DATA_DIR/01_structured/<name>/metadata.{csv,parquet}   # raw metadata symlink
DATA_DIR/02_processed/<name>/metadata.parquet          # cleaned sample-level metadata
DATA_DIR/03_output/<name>/splits/<split_name>.parquet  # canonical tile-level metadata + split labels
DATA_DIR/03_output/<name>/splits/<split_name>/         # full split set from save_splits
```

The split parquet is built by joining `items/all.json` with sample-level metadata on `sample_id`,
replicating sample annotations onto each tile row, then applying `save_splits(...)`.
The canonical tile-level split parquet is the metadata file consumed by `TileDataset`.

Dataset outputs reserve:

```bash
DATA_DIR/03_output/<name>/items/all.json
DATA_DIR/03_output/<name>/panels/default.yaml
```

The `items/` folder is for multiple item-set variants. The `panels/` folder stores YAML files with
`source_panel` and `target_panel` keys.

Download-time structured sample artifacts also include:

```bash
DATA_DIR/01_structured/<name>/<sample_id>/wsi.png
DATA_DIR/01_structured/<name>/<sample_id>/transcripts.png
```

Use the streamed transcript overlay path in `tiling.py`; never load the full raw transcripts parquet
just to render `transcripts.png`.

Training configs use `data.name` to bind to a dataset output root. Relative
`items_path`, `metadata_path`, `panel_path`, and `cache_dir` resolve under `DATA_DIR/03_output/<name>/`.
`panel_path` is resolved under `DATA_DIR/03_output/<name>/panels/` and points to a YAML with `source_panel` and `target_panel`.

## Code philosophy

You are an expert coding assistant for research code in computer vision, following best practices in the field.

- **Zen of Python**: follow these principles explicitly.
  - Beautiful is better than ugly.
  - Explicit is better than implicit.
  - Simple is better than complex.
  - Complex is better than complicated.
  - Flat is better than nested.
  - Sparse is better than dense.
  - Readability counts.
  - Special cases aren't special enough to break the rules.
  - Although practicality beats purity.
  - Errors should never pass silently.
  - Unless explicitly silenced.
  - In the face of ambiguity, refuse the temptation to guess.
  - There should be one-- and preferably only one --obvious way to do it.
  - Although that way may not be obvious at first unless you're Dutch.
  - Now is better than never.
  - Although never is often better than *right* now.
  - If the implementation is hard to explain, it's a bad idea.
  - If the implementation is easy to explain, it may be a good idea.
  - Namespaces are one honking great idea -- let's do more of those!
- **Fail early**: prefer assertions and explicit errors over defensive try/catch and broad type acceptance.
- **Assert assumptions aggressively**: use `assert` regularly to lock in expected data layout, dtypes, categories, shapes, and config invariants.
- **Short assert messages**: keep assertion messages brief and specific, so failures immediately reveal the violated assumption.
- **Assume the research contract, not generality**: do not add flexible handling for broader use cases unless explicitly requested. Assert that inputs match the expected pipeline contract instead.
- **Lean code**: concise, readable, easy to debug. No boilerplate.
- **Python 3.12 typing**: prefer modern built-in typing syntax when possible, including `X | Y` over `Union[X, Y]`.
- **Use libraries**: if a library simplifies the overall codebase, use it.
- **No over-engineering**: this is iterative research code. No extensive error handling, no speculative abstractions, no backwards-compat shims.
- **No trailing summaries**: do not summarize what was just done at the end of a response.

## Preferred tools

Usually prefer these libraries in this project when they fit the task:

- `loguru` for logging
- `jsonargparse` for script CLIs and config-driven entrypoints
- `pathlib` for path handling
- `torch` for modeling and tensor code
- `lightning` for training loops and experiment orchestration

Prefer storing tabular data as parquet unless there is a clear reason to use another format.

## Review style

- Provide comprehensive, direct code reviews.
- Challenge design choices and proactively suggest alternatives or improvements.
- Challenge research questions and approaches; surface what others have done and what alternatives exist.
- Prefer short, direct sentences. Lead with the answer or action.
