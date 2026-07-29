"""
Split ONE sample's own tiles into fit/val/test via sklearn's train_test_split.

Unlike scripts/artifacts/build_splits.py (whole samples hand-assigned to one split each, from
splits/owkin.yaml), this puts tiles from the SAME sample into train, val, and test -- for
single-sample experiments that isolate within-sample structure from cross-sample batch effects.

Output: DATA_DIR/03_output/<name>/splits/<sample_id>/outer=<0..n_folds-1>.parquet -- each fold
an independent random draw -- directly usable as
--train.data.metadata_path <sample_id>/outer=<i>.parquet with the existing training pipeline
(scripts/train/supervised.py) -- no changes needed there.

Usage:
    uv run python scripts/artifacts/build_single_sample_splits.py \\
        --split.name owkin \\
        --split.data_dir $DATA_DIR \\
        --split.items_path c_cells.json \\
        --split.sample_id CH_C_518a_x2 \\
        --split.n_folds 3
"""

from dataclasses import dataclass
from pathlib import Path

from xenium_hne_fusion.artifacts.splits import write_single_sample_splits
from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class SingleSampleSplitConfig:
    name: str = None
    data_dir: Path = None
    items_path: Path = Path("c_cells.json")
    sample_id: str = None
    fit_size: float = 0.6
    val_size: float = 0.1
    test_size: float = 0.3
    n_folds: int = 3
    random_state: int = 0


def main(cfg: SingleSampleSplitConfig, *, overwrite: bool = False) -> int:
    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    items_path = managed.resolve_items_path(cfg.items_path)
    write_single_sample_splits(
        items_path=items_path,
        sample_id=cfg.sample_id,
        output_dir=managed.output_dir,
        fit_size=cfg.fit_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        n_folds=cfg.n_folds,
        random_state=cfg.random_state,
        overwrite=overwrite,
    )
    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(SingleSampleSplitConfig, nested_key="split")
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.split, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
