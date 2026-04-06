---
name: hest1k-panel-overlap
description: Diagnose Xenium gene panel overlap across HEST1K samples and organs, verify whether a training panel is compatible with selected items, and explain mixed-organ panel failures. Use when training crashes on missing genes, when adding organ-specific configs or panels, or when comparing sample and organ gene universes.
---

# HEST1K Panel Overlap

Use this skill when `hest1k` expression training fails because panel genes are missing from `expr-kernel_size=16.parquet`, or when we need to design organ-specific `items`, `splits`, and `panels`.

## Primary check

Use `expr-kernel_size=16.parquet` columns as the source of truth for training compatibility.

- Drop `token_index` if present.
- Do not use `feature_universe.txt` or `transcripts.parquet` categories as the primary check for trainability.
- Use those only as supporting evidence when you need to explain a mismatch.

## Workflow

1. Read the current findings in [references/findings.md](references/findings.md).
2. Run the helper script first:

```bash
.venv/bin/python .codex/skills/hest1k-panel-overlap/scripts/report_overlap.py
```

3. For a specific organ:

```bash
.venv/bin/python .codex/skills/hest1k-panel-overlap/scripts/report_overlap.py --organ Lung
```

4. To verify a panel against selected items:

```bash
.venv/bin/python .codex/skills/hest1k-panel-overlap/scripts/report_overlap.py \
  --items-path data/03_output/hest1k/items/lung.json \
  --panel-path panels/hest1k/hvg-lung-default-outer=0-inner=0-seed=0.yaml
```

## Interpretation

- If a panel is missing genes for any selected sample, the config is not trainable as-is.
- When multiple organs are mixed, report pairwise overlap and the all-organs intersection.
- If gene schemas differ within an organ, report that explicitly. Ignore a lone `token_index` column.
- Prefer organ-specific `items`, `splits`, and `panels` for `hest1k` expression training.

## Repo specifics

- Default `hest1k` items: `data/03_output/hest1k/items/all.json`
- Organ items: `data/03_output/hest1k/items/{breast,lung,pancreas}.json`
- Processed metadata: `data/02_processed/hest1k/metadata.parquet`
- Panels live in `panels/hest1k/`

If you need more context, load [references/findings.md](references/findings.md).
