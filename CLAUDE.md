# xenium-hne-fusion

Research codebase for Xenium spatial transcriptomics and H&E image fusion.

## Project structure

```
xenium-hne-fusion/
├── src/xenium_hne_fusion/   # Importable package (core logic, reusable modules)
├── scripts/                 # One-off analysis and experiment scripts
├── tests/                   # pytest tests
├── data/                    # Managed raw/structured/processed/output data (incl. generated figures)
├── results/                 # Outputs and model artifacts (not tracked)
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
To run python commands use `uv run python`.

### Bash tool PATH
The Bash tool runs with a minimal `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`). Homebrew tools
(e.g. `gh`, `jq`) are not on this path — call them by their full path:

```bash
/opt/homebrew/bin/gh pr create ...
```

```bash
uv sync                  # install all deps incl. dev
uv add <pkg>             # add a runtime dependency
uv add --dev <pkg>       # add a dev dependency
uv run pytest            # run tests
uv run python scripts/my_script.py
```

### Local dependency and search scope

- The editable `ai4bmr-learn` dependency lives at `~/projects/ai4bmr-learn`.
- When worktree-relative dependency resolution breaks, inspect that path directly instead of searching broadly.
- Never search `$HOME`.
- Restrict searches to the current workspace or to an explicit, known dependency path that is directly relevant to the task.

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
Model architecture and training hyperparams live in `configs/train/<dataset>/<task>.yaml`.
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
DATA_DIR/03_output/<name>/splits/<split_name>/         # full split set from save_splits
```

The split parquet is built by joining `items/all.json` with sample-level metadata on `sample_id`,
replicating sample annotations onto each tile row, then applying `save_splits(...)`.
Dataset outputs reserve:

```bash
DATA_DIR/03_output/<name>/items/all.json
panels/<name>/expr.yaml
```

The `items/` folder is for multiple item-set variants. The root-level `panels/` folder stores YAML files with
`source_panel` and `target_panel` keys.

Download-time structured sample artifacts also include:

```bash
DATA_DIR/01_structured/<name>/<sample_id>/wsi.png
DATA_DIR/01_structured/<name>/<sample_id>/transcripts.png
```

Use the streamed transcript overlay path in `tiling.py`; never load the full raw transcripts parquet
just to render `transcripts.png`.

Training configs use `data.name` to bind to a dataset output root. Relative
`items_path`, `metadata_path`, and `cache_dir` resolve under `DATA_DIR/03_output/<name>/`.
`panel_path` is resolved under `panels/<name>/` and points to a YAML with `source_panel` and `target_panel`.

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
- **No script-to-script imports**: scripts under `scripts/` are entrypoints, not reusable modules. Import only from installed packages or `src/xenium_hne_fusion/`. If multiple scripts need the same logic, move that logic into `src/` and keep the scripts as thin wrappers.
- **No over-engineering**: this is iterative research code. No extensive error handling, no speculative abstractions, no backwards-compat shims.
- **No trailing summaries**: do not summarize what was just done at the end of a response.

## Working patterns

- Keep reusable logic in `src/xenium_hne_fusion/`. Keep `scripts/` as thin entrypoints that parse config, call library functions, and return an exit code.
- Use dataclass configs as the boundary between YAML/CLI and implementation. Add fields to the appropriate config dataclass first, then thread the typed config object through the pipeline.
- Prefer shared parser/bridge helpers such as `processing_cli.py` over one-off script parser logic. When jsonargparse adds internal keys, strip them before constructing dataclasses.
- Resolve managed paths through `get_managed_paths(...)`, `build_pipeline_config(...)`, `resolve_training_paths(...)`, and related getters. Do not reconstruct `DATA_DIR/01_structured`, `02_processed`, or `03_output` ad hoc in new code.
- Treat `data.name` as the binding between configs and output roots. Training configs should keep `items_path`, `metadata_path`, `panel_path`, and `cache_dir` relative unless there is a strong local-only reason to use an absolute path.
- Keep artifact names stable and explicit: source items are `items/all.json`; filtered items are `items/<name>.json`; split metadata lives under `splits/<split_name>/`; panels live under the managed output panels directory.
- Use marker files such as `.structured.done` and `.processed.done` to make long data stages idempotent. If an overwrite path needs to reset work, clear the marker and the corresponding processed sample directory together.
- When adding a pipeline stage, structure it as `prepare context -> select/validate samples -> run serial/ray sample work -> finalize dataset`. Keep sample-level functions testable without requiring the full driver.
- Keep panel logic explicit. For expression prediction, require disjoint `source_panel` and `target_panel`; when changing HEST1K organ configs, check the sample gene universe and panel overlap before assuming a mixed-organ panel will work.
- Prefer parquet for tables, JSON for item lists, YAML for configs/panels, and `torch.save`/`torch.load` only for tensor artifacts. Read only the columns needed from large parquet files.
- Use streamed transcript/tile helpers for image overlays and tile-local artifacts. Avoid loading whole raw transcript tables when a tile- or stream-oriented path already exists.
- For training, validate task invariants before building models: `task.target`, `lit.target_key`, output dimensions, required panels, modality encoders, and pooling choices should fail before Lightning starts.
- Keep cache behavior explicit. A missing `cache_dir` means cache disabled; relative cache paths resolve under `DATA_DIR/03_output/<name>/cache/`.

## Tips and tricks

- Use `uv run pytest tests/<test_file>.py` for focused checks before running the full suite.
- For CLI/config changes, add or update small parser bridge tests that import the script with `importlib.util.spec_from_file_location(...)` and assert the resulting dataclass fields.
- For path-resolution changes, test relative and absolute path behavior separately. Most training paths should be relative in YAML and resolved in code.
- For data filtering changes, test the item-level output and the metadata join/split behavior. Sample-level metadata must be replicated onto tile rows through `sample_id`.
- For HEST1K panel or organ changes, use the panel-overlap workflow before training. Missing target/source genes usually mean the selected samples and panel were built from different gene universes.
- For Ray-enabled stages, keep the serial path as the simplest reference implementation. Ray wrappers should submit the same sample-level function and collect failures into a single explicit error.
- When debugging scripts interactively, call the script's `cli([...])` or reusable `main(...)` with explicit arguments instead of relying on global state.
- The shell `PATH` is minimal. Use full paths for Homebrew tools and prefer repo-local commands through `uv run`.
- `cache_dir` (and all training paths) support shell env-var interpolation: set `cache_dir: $TMPDIR/cache` in a YAML config and `_resolve_path` will expand it at runtime via `os.path.expandvars`. Useful for Slurm jobs where `$TMPDIR` is only known at job start.

## Anti-patterns and mistakes to avoid

- Do not import from one script into another script. Move shared code into `src/xenium_hne_fusion/` and import it from there.
- Do not duplicate config parsing across scripts when an existing parser helper can be extended.
- Do not hardcode machine paths, dataset roots, split paths, panel paths, or cache paths in versioned YAML or library code.
- Do not broaden code to accept many alternate schemas unless the pipeline actually needs them. Assert the expected columns, dtypes, categories, shapes, and names instead.
- Do not swallow failed samples, missing genes, missing files, empty item sets, or unknown sample IDs. Fail early with a short assertion or explicit error.
- Do not silently mix tile and patch terminology. A tile is the WSI crop and a patch is a ViT token inside that tile.
- Do not assume a panel that works for one organ works for mixed-organ HEST1K training. Verify intersection/union and sample-level feature universes.
- Do not scan broad home directories to find dependencies or data. Search the workspace or the explicit editable dependency path only.
- Do not add backward-compat shims for old config shapes unless they are already part of the repo contract or explicitly requested.
- Do not add broad `try/except` wrappers around data processing. Let the violated assumption surface.
- Do not use root-level `results/` or ad hoc folders for managed dataset outputs that belong under `DATA_DIR/03_output/<name>/`.
- Do not run full, expensive data/training pipelines as verification unless requested. Prefer targeted unit tests, parser checks, and small debug/fast-dev-run paths.

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
