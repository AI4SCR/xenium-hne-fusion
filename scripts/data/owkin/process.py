"""Process structured owkin samples: normalize modalities, detect tissue, tile, extract per-tile artifacts."""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from xenium_hne_fusion.data.config import DataConfig
from xenium_hne_fusion.data.util import to_pyramidal
from xenium_hne_fusion.processing import extract_tiles, process_cells, process_tiles, tile_cells, tile_transcripts
from xenium_hne_fusion.tiling import detect_tissues, save_points_overview, save_wsi_thumbnail, tile_tissues
from xenium_hne_fusion.utils.getters import ManagedPaths


def normalize_transcripts(transcript_path: Path, output_path: Path) -> None:
    """Keep only predesigned/custom gene transcripts."""
    logger.info(f"Normalizing transcripts: {transcript_path}")
    table = gpd.read_parquet(
        transcript_path,
        columns=["cell_id", "transcript_id", "feature_name", "codeword_category", "geometry"],
    )
    logger.info(f"Codeword categories: {table.codeword_category.value_counts().to_dict()}")
    include = table.codeword_category.isin({"predesigned_gene", "custom_gene"})
    table[include].to_parquet(output_path)


def normalize_cells(cells_path: Path, cell_types_path: Path, cell_type_col: str, output_path: Path) -> None:
    """Collapse tumor subtypes to a single 'tumor' class; store as ordered-categorical GeoDataFrame."""
    cell_types = json.loads(cell_types_path.read_text())
    unique_cell_types = sorted(set(cell_types.values()))
    dtype = pd.CategoricalDtype(unique_cell_types, ordered=True)

    cells = pd.read_parquet(cells_path, columns=["cell_id", "x", "y", cell_type_col])
    cells.loc[:, cell_type_col] = cells[cell_type_col].map(cell_types)
    cells = cells.astype({cell_type_col: dtype})
    cells = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.x, cells.y)).drop(columns=["x", "y"])
    cells.set_index("cell_id").to_parquet(output_path, index=True)


def normalize_wsi(wsi_path: Path) -> None:
    """Convert the structured `wsi.tiff` (a symlink to the raw source) to a tiled, pyramidal TIFF in place.

    Writes to a temp path and atomically swaps it in, so the raw source is never
    written through the symlink. No-op if already converted (no longer a symlink).
    """
    if not wsi_path.is_symlink():
        logger.info(f"Already pyramidal, skipping: {wsi_path}")
        return
    tmp_path = wsi_path.with_suffix(".tmp.tiff")
    to_pyramidal(wsi_path, tmp_path)
    tmp_path.replace(wsi_path)


def prepare_protein_data(cell_features_dir: Path, cells_path: Path, output_path: Path) -> None:
    """Extract protein-expression features and attach cell geometries."""
    import scanpy as sc

    cells = gpd.read_parquet(cells_path)
    ad = sc.read_10x_mtx(cell_features_dir, gex_only=False)

    incl = (ad.var.feature_types == "Protein Expression").values
    proteins = pd.DataFrame(ad.X[:, incl].astype(int).toarray(), index=ad.obs.index, columns=ad.var.index[incl])
    proteins = cells[["geometry"]].merge(proteins, left_index=True, right_index=True)
    proteins.to_parquet(output_path, index=True)


def main(config: DataConfig, *, overwrite: bool = False) -> int:
    paths = ManagedPaths(data_dir=config.data_dir, name=config.name)
    sample_ids = config.filter.select(
        sorted(p.name for p in paths.structured_dir.iterdir() if p.is_dir()),
    )
    tiles = config.tiles

    for sample_id in sample_ids:
        processed_dir = paths.processed_dir / sample_id / f"{tiles.tile_px}_{tiles.stride_px}"
        if processed_dir.exists() and not overwrite:
            logger.info(f"Skipping already-processed {sample_id}")
            continue
        logger.info(f"Processing owkin sample {sample_id}")

        structured_dir = paths.structured_dir / sample_id
        wsi_path = structured_dir / "wsi.tiff"
        transcript_path = structured_dir / "transcripts.parquet"
        transcripts_norm_path = structured_dir / "transcripts_normalized.parquet"
        cells_path = structured_dir / "cells.parquet"
        cells_norm_path = structured_dir / "cells_normalized.parquet"
        protein_path = structured_dir / "proteins.parquet"
        cell_features_dir = structured_dir / "cell_features"
        tissues_path = structured_dir / "tissues.parquet"
        tiles_path = structured_dir / "tiles" / f"{tiles.tile_px}_{tiles.stride_px}.parquet"

        normalize_wsi(wsi_path)
        normalize_transcripts(transcript_path, transcripts_norm_path)
        normalize_cells(cells_path, config.cell_types_path, config.cell_type_col, cells_norm_path)
        prepare_protein_data(cell_features_dir, cells_norm_path, protein_path)

        save_wsi_thumbnail(wsi_path, structured_dir / "wsi.png")
        save_points_overview(wsi_path, transcripts_norm_path, structured_dir / "transcripts.png")
        save_points_overview(wsi_path, cells_norm_path, structured_dir / "cells.png")

        detect_tissues(wsi_path, tissues_path)
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tile_tissues(
            wsi_path,
            tissues_parquet=tissues_path,
            tile_px=tiles.tile_px,
            stride_px=tiles.stride_px,
            mpp=tiles.mpp,
            output_parquet=tiles_path,
        )

        tiles_gdf = gpd.read_parquet(tiles_path)
        extract_tiles(wsi_path, tiles_gdf, processed_dir, tiles.mpp, img_size=tiles.img_size)
        tile_transcripts(
            tiles=tiles_gdf,
            transcripts_path=transcripts_norm_path,
            output_dir=processed_dir,
            img_size=tiles.img_size,
            predicate=tiles.predicate,
        )
        process_tiles(tiles_gdf, processed_dir, img_size=tiles.img_size, kernel_size=tiles.kernel_size)

        tile_cells(tiles_gdf, cells_norm_path, processed_dir, name="cells.parquet", predicate=tiles.predicate)
        process_cells(tiles_gdf, processed_dir, img_size=tiles.img_size, cell_type_col=None, normalize=False, name="cells.parquet")

        tile_cells(tiles_gdf, protein_path, processed_dir, name="proteins.parquet", predicate=tiles.predicate)
        process_cells(tiles_gdf, processed_dir, img_size=tiles.img_size, cell_type_col=None, normalize=False, name="proteins.parquet")
    return 0


def cli(argv: list[str] | None = None) -> int:
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True)
    parser.add_class_arguments(DataConfig, nested_key="data")
    parser.add_argument("--overwrite", type=bool, default=False)

    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.data, overwrite=init.overwrite)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
