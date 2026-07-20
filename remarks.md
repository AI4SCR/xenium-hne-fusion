# Remarks

## PROTEIN_PANEL naming mismatch is item-set specific (g_cells vs c_cells/d_cells)

`PROTEIN_PANEL` in `src/xenium_hne_fusion/targets.py` lists `'PCNA'` and `'PTEN-1'`.

For the **g_cells** `proteins.parquet` files, the actual columns are named the other way
around: `'PCNA-1'` and `'PTEN'` (no `-1` on PTEN, `-1` suffix on PCNA instead). This causes
`pd.read_parquet(tile_dir / 'proteins.parquet', columns=PROTEIN_PANEL)` in
`src/xenium_hne_fusion/datasets/tiles.py:114` to fail with
`pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(PCNA)` during a debug run on g_cells.

This mismatch does **not** apply to the `c_cells`/`d_cells` item sets — their
`proteins.parquet` column naming matches `PROTEIN_PANEL` as currently written.

Do not blanket-fix `PROTEIN_PANEL` to the g_cells naming — that would break c_cells/d_cells.
Needs a per-item-set resolution (e.g. verify actual column names per item set before deciding
whether to normalize upstream at data-generation time or branch the column list per `data.name`).
