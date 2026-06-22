
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import geopandas as gpd
from xenium_hne_fusion.config import DataConfig

from xenium_hne_fusion.utils.getters import build_pipeline_config
from xenium_hne_fusion.processing_cli import parse_data_args
import yaml
from xenium_hne_fusion.utils.getters import DEFAULT_CELL_TYPE_COL


# TODO
def main(data_cfg: DataConfig) -> None:
    load_dotenv()
    assert data_cfg.name == "owkin", f"Expected dataset='owkin', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)

    sample_dirs = sorted(p for p in cfg.raw_dir.iterdir() if p.is_dir())
   
    logger.info(f"Found {len(sample_dirs)} samples in {cfg.raw_dir}")


    with open("cell_types/owkin/cell_types.yaml") as f:
        config = yaml.safe_load(f)

    cell_types = config["cell_types"]
    celltype_dtype = pd.CategoricalDtype(categories=cell_types, ordered=True)

    for sample_dir in sample_dirs:
        cells_path = sample_dir / "cells_.parquet"
        new_cells_path = sample_dir / "cells.parquet"

        cells = gpd.read_parquet(cells_path)
        cells.set_geometry(
            gpd.points_from_xy(cells.x, cells.y), 
            inplace=True,
        )

        # add categorical cell types column
        cells[DEFAULT_CELL_TYPE_COL] = cells["first_type"].astype(celltype_dtype)
        cells.to_parquet(new_cells_path)
        logger.info(f"Saved new file at {new_cells_path}")

if __name__ == "__main__":
    data_cfg, _, _, _ = parse_data_args(['--config', 'configs/data/remote/beat.yaml'], include_executor=False)
    main(data_cfg)
