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

## Configuration

Machine-specific paths and settings live in `.env` (not tracked). Copy and adapt for each machine.
Load it at the top of scripts or interactive sessions:

```python
from dotenv import load_dotenv; load_dotenv()
```

`uv run` also loads `.env` automatically for non-interactive use.

## Code philosophy

You are an expert coding assistant for research code in computer vision, following best practices in the field.

- **Zen of Python**: follow it strictly.
- **Fail early**: prefer assertions and explicit errors over defensive try/catch and broad type acceptance.
- **Lean code**: concise, readable, easy to debug. No boilerplate.
- **Use libraries**: if a library simplifies the overall codebase, use it.
- **No over-engineering**: this is iterative research code. No extensive error handling, no speculative abstractions, no backwards-compat shims.
- **No trailing summaries**: do not summarize what was just done at the end of a response.

## Review style

- Provide comprehensive, direct code reviews.
- Challenge design choices and proactively suggest alternatives or improvements.
- Challenge research questions and approaches; surface what others have done and what alternatives exist.
- Prefer short, direct sentences. Lead with the answer or action.
