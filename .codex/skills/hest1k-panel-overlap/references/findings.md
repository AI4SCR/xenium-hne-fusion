# HEST1K overlap findings

Primary training compatibility check:

- Use `expr-kernel_size=16.parquet` column names.
- Ignore `token_index` if present.
- Treat `feature_universe.txt` and `transcripts.parquet` categories as supporting evidence.

Current repo findings:

- Breast sample `NCBI783`: 288 genes
- Lung sample `NCBI856`: 343 genes
- Pancreas sample `TENX116`: 474 genes
- Breast vs Lung overlap: 83 genes
- Breast vs Pancreas overlap: 137 genes
- Lung vs Pancreas overlap: 111 genes
- All-organs intersection: 58 genes
- All-organs union: 832 genes

Current mixed `hest1k` panel:

- Path: `panels/hest1k/hvg-default-default-outer=0-inner=0-seed=0.yaml`
- Total genes: 280
- Compatible with Breast
- Incompatible with Lung
- Incompatible with Pancreas

Observed schema drift inside an organ:

- The only intra-organ schema difference found so far is an extra `token_index` column on some tiles.
- After dropping `token_index`, gene columns are stable within each current organ.

Implication:

- Mixed-organ `hest1k` expression training is brittle.
- Organ-specific `items`, `splits`, and `panels` are the safe path.
