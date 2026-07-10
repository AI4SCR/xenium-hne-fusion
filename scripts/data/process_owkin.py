"""Process BEAT samples: tissue detection, tiling, transcript and cell extraction."""

import sys

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from xenium_hne_fusion.config import DataConfig
from xenium_hne_fusion.processing import extract_tiles, process_cells, process_tiles, tile_cells, tile_transcripts
from xenium_hne_fusion.processing_cli import parse_data_args
from xenium_hne_fusion.tiling import detect_tissues, tile_tissues
from xenium_hne_fusion.utils.getters import build_pipeline_config, select_sample_ids

from pathlib import Path

import pandas as pd
import geopandas as gpd
# cells_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/cells.parquet')
# cells = gpd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/cells.parquet', columns=['cell_id', 'x', 'centroid_x', 'geometry'])
# cells = gpd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/cells.parquet')
# transcripts = gpd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/transcripts.parquet', columns=['cell_id', 'x_location', 'geometry'])
# cell_id = 'bkadoedk-1'
# cell = cells[cells.cell_id == cell_id]
# cell_tx = transcripts[transcripts.cell_id == cell_id]
# cell_tx.x_location.mean()
# cell_tx.geometry.x.mean()

# data_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/owkin_spatial/data/crc_gbm_dlbcl_pilot/xenium_multimodal/results/matched_samples_cells_aligned')
# sample_dirs = [d for d in data_dir.glob('*') if d.is_dir()]
# cell_types = set()
# for sample_dir in sample_dirs:
#     cells = pd.read_parquet(sample_dir / 'centroids' / 'centroid_cropped.parquet', columns=['first_type'])
#     tumor_cell = cells.first_type.str.startswith('Tu_CH_')
#     cells.loc[tumor_cell, 'first_type'] = 'tumor'
#     cell_types.update(cells.first_type.unique())
# sorted(cell_types)


def prepare_protein_data(cell_features_dir: Path, cells_path: Path, output_path: Path):
    import scanpy as sc
    import geopandas as gpd

    cells = gpd.read_parquet(cells_path)

    # cell_features_dir = '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/cell_features'
    ad = sc.read_10x_mtx(cell_features_dir, gex_only=False)

    # incl = (ad.var.feature_types == 'Gene Expression').values
    # transcripts = pd.DataFrame(ad.X[:, incl].astype(int).toarray(), index=ad.obs.index, columns=ad.var.index[incl])

    incl = (ad.var.feature_types == 'Protein Expression').values
    proteins = pd.DataFrame(ad.X[:, incl].astype(int).toarray(), index=ad.obs.index, columns=ad.var.index[incl])

    proteins = cells[['geometry']].merge(proteins, left_index=True, right_index=True)
    proteins.to_parquet(output_path, index=True)

def normalize_cells(cells_path: Path, output_path: Path):
    import yaml
    cell_types = yaml.load(Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/cell_types/owkin/cell_types.yaml').open(), Loader=yaml.SafeLoader)
    dtype = pd.CategoricalDtype(cell_types['cell_types'], ordered=True)
    # cells = gpd.read_parquet(cells_path, columns=['cell_id', 'x', 'y', 'first_type', 'scond_type', 'geometry'])
    cells = pd.read_parquet(cells_path, columns=['cell_id', 'x', 'y', 'first_type'])
    tumor_cell = cells.first_type.str.startswith('Tu_CH_')
    cells.loc[tumor_cell, 'first_type'] = 'tumor'
    cells = cells.astype({'first_type': dtype})
    cells = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.x, cells.y)).drop(columns=['x', 'y'])
    cells.set_index('cell_id').to_parquet(output_path)

def normalize_transcripts(transcript_path: Path, output_path: Path):

    logger.info(f'Normalizing transcript file: {transcript_path}')
    table = gpd.read_parquet(transcript_path, columns=['cell_id', 'transcript_id', 'feature_name', 'codeword_category', 'geometry'])
    # table = table.sample(10_000)
    include = table.codeword_category.isin({'predesigned_gene', 'custom_gene'})
    codewords = table.codeword_category.value_counts().to_dict()
    logger.info(f'{codewords}')

    table = table[include]

    # geometry = gpd.points_from_xy(x=table.x_location, y=table.y_location)
    # table = gpd.GeoDataFrame(table, geometry=geometry)
    table.to_parquet(output_path)

def to_ome_zarr(wsi_path: Path, output_path: Path):
    from pathlib import Path
    from ome_zarr import OMEZarrImage, OMEZarrMultiscale

    import dask.array as da
    import tifffile
    import zarr
    import dask.array as da

    src = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/wsi.tiff"
    dst = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/owkin/CH_C_518a_x2/wsi.ome.zarr"

    with tifffile.TiffFile(src) as tif:
        assert len(tif.pages) == 1

        page = tif.pages[0]

        xres = page.tags["XResolution"].value  # (numerator, denominator)
        x_pixels_per_unit = xres[0] / xres[1]

        yres = page.tags["YResolution"].value
        y_pixels_per_unit = yres[0] / yres[1]

        unit = page.tags["ResolutionUnit"].value
        if unit.name == 'CENTIMETER':
            um_per_x_pixel = 10_000 / x_pixels_per_unit
            um_per_y_pixel = 10_000 / y_pixels_per_unit
        else:
            raise ValueError('Expected unit to be CENTIMETER')


    # Lazily expose tiled TIFF pages as zarr arrays
    Path(src).exists()

    tiff_store = tifffile.imread(src, aszarr=True)
    root = zarr.open(tiff_store, mode="r")
    arr = da.from_array(root, chunks=(1024, 1024, 3))
    arr = arr.transpose(2, 0, 1)

    image = OMEZarrImage(data=arr,
                         axes=['c', 'y', 'x'],
                         scale={"c": 1.0, "y": um_per_y_pixel, "x": um_per_x_pixel},
                         axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"}
                         )

    multiscale = OMEZarrMultiscale(image=image, scale_factors=(2, 4, 8), method="resize")
    multiscale.to_ome_zarr(dst, overwrite=True)


def main(
    data_cfg: DataConfig,
    overwrite: bool = False,
) -> None:
    # from wsidata import

    assert data_cfg.name in ["owkin"], f"Expected dataset=owkin', got {data_cfg.name!r}"
    cfg = build_pipeline_config(data_cfg)
    sample_ids = select_sample_ids(
        sorted(p.name for p in cfg.paths.structured_dir.iterdir() if p.is_dir()),
        cfg.data.filter,
    )
    tiles_cfg = cfg.data.tiles
    assert tiles_cfg.img_size is not None, "tiles.img_size is required"
    img_size = tiles_cfg.img_size
    kernel_size = tiles_cfg.kernel_size
    predicate = tiles_cfg.predicate

    for sample_id in sample_ids:
        # break
        logger.info(f"Processing BEAT sample {sample_id}")
        structured_dir = cfg.paths.structured_dir / sample_id
        wsi_path = structured_dir / "wsi.tiff"
        # ome_zarr_path = structured_dir / "wsi.ome.zarr"
        transcript_path = structured_dir / "transcripts.parquet"
        transcripts_norm_path = structured_dir / "transcripts_normalized.parquet"
        cells_path = structured_dir / "cells.parquet"
        cells_norm_path = structured_dir / "cells_normalized.parquet"
        protein_path = structured_dir / "proteins.parquet"
        cell_features_dir = structured_dir / 'cell_features'
        tissues_path = structured_dir / "tissues.parquet"
        tiles_path = structured_dir / "tiles" / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}.parquet"
        processed_dir = cfg.paths.processed_dir / sample_id / f"{tiles_cfg.tile_px}_{tiles_cfg.stride_px}"

        normalize_transcripts(transcript_path=transcript_path, output_path=transcripts_norm_path)
        normalize_cells(cells_path=cells_path, output_path=cells_norm_path)
        prepare_protein_data(cell_features_dir=cell_features_dir, cells_path=cells_norm_path, output_path=protein_path)
        # to_ome_zarr(wsi_path=wsi_path, output_path=ome_zarr_path)

        detect_tissues(wsi_path, tissues_path)
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tile_tissues(
            wsi_path,
            tissues_parquet=tissues_path,
            tile_px=tiles_cfg.tile_px,
            stride_px=tiles_cfg.stride_px,
            mpp=tiles_cfg.mpp,
            output_parquet=tiles_path,
        )
        tiles = gpd.read_parquet(tiles_path)
        extract_tiles(wsi_path, tiles, processed_dir, tiles_cfg.mpp, img_size=img_size)
        tile_transcripts(tiles=tiles, transcripts_path=transcripts_norm_path, output_dir=processed_dir, img_size=img_size, predicate=predicate)
        process_tiles(tiles, processed_dir, img_size=img_size, kernel_size=kernel_size)
        if cells_path.exists():
            tile_cells(tiles, cells_norm_path, processed_dir, name='cells.parquet', predicate=predicate)
            process_cells(tiles, processed_dir, img_size=img_size, cell_type_col=None, normalize=False, name='cells.parquet')

        if protein_path.exists():
            tile_cells(tiles, cells_path=protein_path, output_dir=processed_dir, predicate=predicate, name='proteins.parquet')
            process_cells(tiles, processed_dir, img_size=img_size, cell_type_col=None, normalize=False, name='proteins.parquet')


def cli(argv: list[str] | None = None) -> int:
    data_cfg, overwrite, _, _ = parse_data_args(argv, include_executor=False)
    main(data_cfg, overwrite=overwrite)
    return 0

# %%
if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))