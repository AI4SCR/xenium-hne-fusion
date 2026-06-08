import lazyslide as zs
import geopandas as gpd
from skimage import io

wsi = zs.open_wsi(wsi='/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/beat/XE_1CGZ.01_42_HNE_1CGZ/wsi.tiff')
tiles = gpd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/01_structured/beat/XE_1CGZ.01_42_HNE_1CGZ/tiles/512_256.parquet')
tile = tiles[tiles.tile_id == 0].squeeze()
x, y, w, h = tile.x_px, tile.y_px, tile.width_px, tile.height_px
img = wsi.reader.get_region(x, y, w, h, level=0)  # (H, W, 3) uint8

io.imsave('/work/FAC/FBM/DBC/mrapsoma/prometex/data/img.png', img)

img_size = 224
img_resized = wsi.reader.resize_img(img, dsize=(img_size, img_size))
io.imsave('/work/FAC/FBM/DBC/mrapsoma/prometex/data/img_resize.png', img_resized)


