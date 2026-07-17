"""Process structured hest1k samples.

NOTE: hest1k raw transcripts carry `he_x`/`he_y` coordinate columns, not `geometry`.
The shared pipeline (`xenium_hne_fusion.tiling.save_points_overview` and friends) now
assumes a `geometry` column unconditionally. Before this script can run the shared
tiling/overview pipeline, it must normalize transcripts by converting `he_x`/`he_y`
into a `geometry` column, e.g.:

    transcripts["geometry"] = gpd.points_from_xy(transcripts["he_x"], transcripts["he_y"])

This normalization step is not yet implemented here — hest1k processing is not
migrated to the new pipeline yet (see owkin's `structure_owkin.py`/`process_owkin.py`
for the target shape once this is filled in).
"""
