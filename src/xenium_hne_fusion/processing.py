from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from loguru import logger
from shapely.geometry import box
from wsidata import open_wsi

BIOLOGICAL_FEATURE_EXCLUDE_PREFIXES = (
    "BLANK_",
    "NegControlCodeword_",
    "NegControlProbe_",
    "UnassignedCodeword_",
    "DeprecatedCodeword_",
    "Intergenic_Region_",
)


def extract_tiles(
    wsi_path: Path,
    tiles: gpd.GeoDataFrame,
    output_dir: Path,
    mpp: float,
) -> None:
    """
    Crop and save a tile.pt for each tile.

    Reads native-resolution region, resizes to tile_px × tile_px at target mpp,
    saves as uint8 CHW torch tensor.
    """
    from PIL import Image

    wsi = open_wsi(wsi_path)
    native_mpp = wsi.properties.mpp
    assert native_mpp is not None, "WSI has no mpp metadata"

    logger.info(f"Extracting {len(tiles)} tiles (native mpp={native_mpp:.4f}, target mpp={mpp})")
    for _, tile in tiles.iterrows():
        tile_dir = output_dir / str(tile.tile_id)
        tile_dir.mkdir(parents=True, exist_ok=True)

        x, y, w, h = tile.x_px, tile.y_px, tile.width_px, tile.height_px
        tile_px = round(w * native_mpp / mpp)

        img = wsi.reader.get_region(x, y, w, h, level=0)  # (H, W, 3) uint8
        img = Image.fromarray(img).resize((tile_px, tile_px), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # CHW
        torch.save(tensor, tile_dir / "tile.pt")

    logger.info("Tile extraction done")


def filter_transcripts(transcripts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove control-probe transcripts by feature_name prefix."""
    transcripts = transcripts.copy()
    transcripts["feature_name"] = normalize_feature_names(transcripts["feature_name"])
    mask = transcripts.feature_name.str.startswith(BIOLOGICAL_FEATURE_EXCLUDE_PREFIXES)
    return transcripts[~mask]


def normalize_feature_names(feature_names: pd.Series) -> pd.Series:
    if len(feature_names) == 0:
        return feature_names
    if isinstance(feature_names.iloc[0], bytes):
        return feature_names.str.decode("utf-8")
    return feature_names


def infer_feature_universe(
    transcripts_path: Path,
    *,
    feature_universe_path: Path | None = None,
    batch_size: int = 1_000_000,
) -> list[str]:
    if feature_universe_path is not None and feature_universe_path.exists():
        return load_feature_universe(feature_universe_path)

    transcripts = pq.ParquetFile(transcripts_path)
    features: set[str] = set()

    for batch in transcripts.iter_batches(batch_size=batch_size, columns=["feature_name"]):
        chunk = batch.to_pandas()
        feature_names = normalize_feature_names(chunk["feature_name"])
        keep = ~feature_names.str.startswith(BIOLOGICAL_FEATURE_EXCLUDE_PREFIXES)
        features.update(feature_names.loc[keep].unique().tolist())

    feature_universe = sorted(features)
    assert feature_universe, f"No biological features found in {transcripts_path}"

    if feature_universe_path is not None:
        save_feature_universe(feature_universe, feature_universe_path)

    return feature_universe


def save_feature_universe(feature_universe: list[str], feature_universe_path: Path) -> None:
    feature_universe_path.parent.mkdir(parents=True, exist_ok=True)
    feature_universe_path.write_text("\n".join(feature_universe) + "\n")


def load_feature_universe(feature_universe_path: Path) -> list[str]:
    return [line for line in feature_universe_path.read_text().splitlines() if line]


def set_feature_universe(feature_names: pd.Series, feature_universe: list[str]) -> pd.Series:
    normalized = normalize_feature_names(feature_names)
    categorical = pd.Categorical(normalized, categories=feature_universe, ordered=False)
    return pd.Series(categorical, index=feature_names.index, name=feature_names.name)


def tile_transcripts(tiles, transcripts_path: Path, save_dir: Path, predicate: str = "within") -> None:
    # NOTE: this will only create parquet files for tiles that have transcripts!

    transcripts = pq.ParquetFile(transcripts_path)

    logger.info(
        f"Tiling transcripts (num_tiles={len(tiles)}, num_transcripts={transcripts.metadata.num_rows})..."
    )

    chunk_size = 1_000_000
    num_chunks = transcripts.metadata.num_rows // chunk_size + 1
    for j, batch in enumerate(
        transcripts.iter_batches(
            batch_size=chunk_size,
            columns=["transcript_id", "cell_id", "feature_name", "geometry"],
        ),
        start=1,
    ):
        logger.info(f"Processing chunk {j}/{num_chunks}")
        chunk = gpd.GeoDataFrame.from_arrow(batch)
        chunk = filter_transcripts(chunk)

        # NOTE: alternative predicates: 'intersects'.
        joined = gpd.sjoin(chunk, tiles, how="inner", predicate=predicate)
        joined = joined.drop(columns=["index_right"]).to_arrow()

        ds.write_dataset(
            data=joined,
            base_dir=str(save_dir),
            format="parquet",
            basename_template=f"part-{{i}}-chunk={j}.parquet",
            partitioning=["tile_id"],
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
        )

        del chunk, joined
        gc.collect()


def get_patchified_transcripts(tile_id: int, transcripts_dir: Path) -> gpd.GeoDataFrame | None:
    """
    Load transcripts for a single tile from the hive-partitioned dataset.

    Returns None if the tile has no transcripts.
    Reconstructs geometry from WKB so the result is a proper GeoDataFrame.
    """
    tile_dir = transcripts_dir / f"tile_id={tile_id}"
    if not tile_dir.exists():
        return None

    df = pd.read_parquet(tile_dir)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geometry))


def make_token_tiles(img_size: int, kernel_size: int) -> gpd.GeoDataFrame:
    """Non-overlapping square token tiles in [0, img_size] image coords, top→bottom left→right."""
    assert img_size % kernel_size == 0, f"{img_size} not divisible by {kernel_size}"
    n = img_size // kernel_size
    geoms = [
        box(c * kernel_size, r * kernel_size, (c + 1) * kernel_size, (r + 1) * kernel_size)
        for r in range(n)
        for c in range(n)
    ]
    return gpd.GeoDataFrame({"tile_id": range(len(geoms))}, geometry=geoms)


def process_tiles(
    tiles: gpd.GeoDataFrame,
    transcripts_dir: Path,
    output_dir: Path,
    raw_transcripts_path: Path,
    img_size: int = 256,
    kernel_size: int = 16,
) -> None:
    from skimage.io import imsave

    from ai4bmr_learn.plotting.patches import draw_tiles
    from ai4bmr_learn.plotting.xenium import visualize_points

    token_tiles = make_token_tiles(img_size, kernel_size)
    feature_universe_path = output_dir.parent / "feature_universe.txt"
    feature_universe = infer_feature_universe(
        raw_transcripts_path,
        feature_universe_path=feature_universe_path,
    )
    logger.info(f"Processing {len(tiles)} tiles (img_size={img_size}, kernel_size={kernel_size})")

    for _, tile in tiles.iterrows():
        tile_id = tile.tile_id
        tile_dir = output_dir / str(tile_id)

        pts = get_patchified_transcripts(tile_id, transcripts_dir)
        if pts is None:
            continue  # no transcripts in this tile

        pts = transform_points(pts, tile, dst_height=img_size, dst_width=img_size, errors="clip_warn")
        pts = pts.drop(columns="index_right", errors="ignore")
        pts["feature_name"] = set_feature_universe(pts["feature_name"], feature_universe)
        pts.to_parquet(tile_dir / "transcripts.parquet")

        expr = compute_expr_tokens(
            pts,
            tiles=token_tiles,
            feature_universe=feature_universe,
            group_by="feature_name",
        )
        expr.columns = expr.columns.astype(str)
        expr.to_parquet(tile_dir / f"expr-kernel_size={kernel_size}.parquet")

        img = torch.load(tile_dir / "tile.pt").permute(1, 2, 0).numpy()
        imsave(tile_dir / "tile.png", img)

        top_feats = pts.feature_name.value_counts().index[:5].tolist()
        viz = visualize_points(pts, image=img.copy(), radius=1)
        viz = draw_tiles(viz, token_tiles, thickness=1, alpha=0.7)
        imsave(tile_dir / "transcripts.png", viz)

        viz = visualize_points(pts, image=img.copy(), radius=1, color_by_label=True, include_labels=top_feats, legend=True)
        viz = draw_tiles(viz, token_tiles, thickness=1, alpha=0.7)
        imsave(tile_dir / "transcripts_top5_feats.png", viz)

    logger.info("Tile processing done")


def transform_points(
    points: gpd.GeoSeries,
    tile: gpd.GeoSeries,
    dst_height: int,
    dst_width: int,
    subset: np.ndarray | None = None,
    errors: str = "raise",
):
    pts = points
    subset = subset if subset is not None else np.ones(len(pts), dtype=bool)

    xmin, ymin, xmax, ymax = tile.geometry.bounds
    height = ymax - ymin
    width = xmax - xmin

    xs = pts.geometry.x.to_numpy(copy=False)
    ys = pts.geometry.y.to_numpy(copy=False)

    xoff, yoff = -xmin, -ymin
    xfact = dst_width / width
    yfact = dst_height / height

    xs_m = (xs[subset] + xoff) * xfact
    ys_m = (ys[subset] + yoff) * yfact

    validate_points(xs=xs_m, ys=ys_m, height=dst_height, width=dst_width, errors=errors)

    pts["geometry"] = gpd.points_from_xy(xs_m, ys_m)
    return pts


def validate_points(xs, ys, height: int, width: int, errors: str = "raise"):
    if "clip" in errors:
        xs = np.clip(xs, 0, width)
        ys = np.clip(ys, 0, height)

    if "raise" in errors or "warn" in errors:
        atol = 1e-6
        xmin, ymin = xs.min(), ys.min()
        xmax, ymax = xs.max(), ys.max()

        if xmin < 0:
            condition = np.isclose(xmin, 0, rtol=0, atol=atol)
            raise_or_warn(condition=condition, msg=f"xmin={xmin} < 0", errors=errors)
        if ymin < 0:
            condition = np.isclose(ymin, 0, rtol=0, atol=atol)
            raise_or_warn(condition=condition, msg=f"ymin={ymin} < 0", errors=errors)
        if xmax > width:
            condition = np.isclose(xmax, width, rtol=0, atol=atol)
            raise_or_warn(condition=condition, msg=f"xmax={xmax} > {width}", errors=errors)
        if ymax > height:
            condition = np.isclose(ymax, height, rtol=0, atol=atol)
            raise_or_warn(condition=condition, msg=f"ymax={ymax} > {height}", errors=errors)


def raise_or_warn(condition: bool, msg: str, errors: str):
    if not condition:
        if "warn" in errors:
            logger.warning(msg)
        else:
            raise ValueError(msg)


def generate_xenium_subsets(
    *,
    points: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    col_name: str,
    predicate: str = "intersects",
    allow_duplicates: bool = False,
    remove_duplicates: bool = True,
    id_col: str = "transcript_id",
):
    """
    Assigns Xenium transcripts to spatial tiles via a spatial join.

    Args:
        points: GeoDataFrame of transcript point geometries.
        tiles: GeoDataFrame of tile polygons.
        col_name: Column name to store the tile index assignment.
        predicate: Spatial predicate ('within' or 'intersects'). Default 'intersects'.
        allow_duplicates: If False, raises when a transcript maps to multiple tiles.
        remove_duplicates: If True, drops duplicate transcript IDs after join.
    """

    points = gpd.sjoin(points, tiles, predicate=predicate)
    points.rename(columns={"index_right": col_name}, inplace=True)

    if not allow_duplicates:
        assert not points[id_col].duplicated().any()

    if remove_duplicates:
        filter_ = points[id_col].duplicated()
        points = points[~filter_]

    return points


def compute_expr_tokens(
    points: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    feature_universe: list[str],
    group_by: str = "feature_name",
    id_col: str = "transcript_id",
) -> pd.DataFrame:
    """Compute token-level expression counts by aggregating transcripts within each token tile.

    Args:
        points: GeoDataFrame of transcripts in tile-local image coords.
        tiles: GeoDataFrame of token tile polygons (from make_token_tiles).
        group_by: Column to accumulate counts by (e.g. 'feature_name').
        id_col: Column uniquely identifying each transcript.

    Returns:
        DataFrame with token_index as index and group_by values as columns.
    """
    subsets = generate_xenium_subsets(
        points=points,
        tiles=tiles,
        col_name="token_index",
        predicate="intersects",
        allow_duplicates=True,
        remove_duplicates=True,
        id_col=id_col,
    )
    assert len(subsets) == len(points)

    subsets[group_by] = pd.Categorical(subsets[group_by], categories=feature_universe, ordered=False)
    tokens = expr_pool(subsets, num_tokens=len(tiles), group_by=group_by)
    tokens = tokens.reindex(columns=feature_universe, fill_value=0)
    assert not tokens.isna().any().any(), "Expression data contains NaN values after pooling."
    return tokens


def expr_pool(
    expr_subsets: gpd.GeoDataFrame,
    pooling: str = "cnt",
    num_tokens: int | None = None,
    group_by: str = "ensembl_id",
) -> pd.DataFrame:
    """Pool expression data to the token level.
    Args:
        expr_subsets: all transcripts for one patch
        pooling: pooling strategy
        num_tokens: the number of tokens to return
        group_by: the column to accumulate the counts by
    Returns:
        token level expression data: dataframe with token_index as index and group_by as columns
    """
    assert expr_subsets[group_by].dtype == "category", f"{group_by} must be categorical"

    if len(expr_subsets) == 0:
        num_feat = expr_subsets[group_by].dtype.categories.size  # pyright: ignore[reportAttributeAccessIssue]
        columns = expr_subsets[group_by].dtype.categories  # pyright: ignore[reportAttributeAccessIssue]
        return pd.DataFrame(np.zeros((num_tokens or 0, num_feat)), columns=columns).astype(int)

    if pooling in ["cnt"]:
        tokens = expr_subsets.groupby(["token_index", group_by], observed=False).size().unstack().fillna(0).astype(int)
    else:
        raise NotImplementedError(f"{pooling} is not implemented.")

    if num_tokens:
        if len(tokens) == num_tokens:
            return tokens
        return tokens.reindex(range(num_tokens), fill_value=0)
    return tokens
