#!/usr/bin/env bash
set -euo pipefail

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=bowel-expression-expr-tile-0 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/bowel/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/bowel/outer=0-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/bowel-hvg-outer=0-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-token-0 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=0-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=0-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=lung-expression-expr-token-0 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/lung/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/lung/outer=0-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/lung-hvg-outer=0-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=bowel-expression-expr-tile-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/bowel/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/bowel/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/bowel-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=bowel-expression-expr-token-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/bowel/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/bowel/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/bowel-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-token-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=lung-expression-expr-tile-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/lung/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/lung/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/lung-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=pancreas-expression-expr-token-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/pancreas/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/pancreas/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/pancreas-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=bowel-expression-expr-tile-2 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/bowel/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/bowel/outer=2-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/bowel-hvg-outer=2-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=bowel-expression-expr-token-2 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/bowel/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/bowel/outer=2-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/bowel-hvg-outer=2-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-tile-2 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=2-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-token-2 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=2-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=2-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=lung-expression-expr-tile-2 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/lung/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/lung/outer=2-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/lung-hvg-outer=2-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-tile-3 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=3-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=3-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-expr-token-3 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/expr-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=3-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=3-inner=0-seed=0.yaml --backbone.learnable_gate false"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-early-fusion-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/early-fusion.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate true"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-late-fusion-token-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/late-fusion-token.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate true"

sbatch \
    --cpus-per-task=12 \
    --mem=64G \
    --gres=gpu:1 \
    --partition=gpu-l40 \
    --time=04:00:00 \
    --output=$HOME/logs/%j.out \
    --job-name=breast-expression-late-fusion-tile-1 \
    --wrap="uv run python scripts/train/supervised.py --config configs/train/hest1k/expression/breast/late-fusion-tile.yaml --data.metadata_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/splits/breast/outer=1-inner=0-seed=0.parquet --data.panel_path /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/hest1k/panels/breast-hvg-outer=1-inner=0-seed=0.yaml --backbone.learnable_gate true"
