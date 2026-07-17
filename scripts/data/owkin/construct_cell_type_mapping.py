from pathlib import Path
import pandas as pd
import json

raw_dir = Path(
    '/work/PRTNR/CHUV/DIR/rgottar1/owkin_spatial/data/crc_gbm_dlbcl_pilot/xenium_multimodal/results/matched_samples_cells_aligned')
cell_paths = sorted(raw_dir.rglob('centroid_cropped.parquet'))

cell_types = None
for cell_path in cell_paths:
    cells = pd.read_parquet(cell_paths, columns=['first_type'])
    if cell_types is None:
        cell_types = set(cells.first_type.unique())
    else:
        assert cell_types == set(cells.first_type.unique())

cell_type_mapping = {x: 'tumor' if x.startswith('Tu_CH_') else x for x in cell_types}

save_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/cell_types/owkin/cell_types.json')
save_path.parent.mkdir(parents=True, exist_ok=True)

json.dump(cell_type_mapping, fp=save_path.open('w'))
