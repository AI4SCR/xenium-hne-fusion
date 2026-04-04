import pytest
from shapely.geometry import box

from xenium_hne_fusion.processing import make_token_tiles


def test_output_length():
    gdf = make_token_tiles(256, 16)
    assert len(gdf) == 256


def test_tile_ordering_first_second_and_17th():
    gdf = make_token_tiles(256, 16)
    bounds = [geom.bounds for geom in gdf.geometry]  # (minx, miny, maxx, maxy)
    assert bounds[0] == (0, 0, 16, 16)
    assert bounds[1] == (16, 0, 32, 16)
    assert bounds[16] == (0, 16, 16, 32)


def test_all_tiles_are_square_with_correct_side():
    gdf = make_token_tiles(256, 16)
    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        assert maxx - minx == 16
        assert maxy - miny == 16


def test_tiles_exactly_cover_image():
    img_size, kernel_size = 256, 16
    gdf = make_token_tiles(img_size, kernel_size)
    union = gdf.geometry.union_all()
    full_square = box(0, 0, img_size, img_size)
    assert union.equals(full_square)


def test_non_divisible_img_size_raises():
    with pytest.raises(AssertionError):
        make_token_tiles(255, 16)
