'''
This script requires patching the UpSet library install from NirOfir:fix-chainedassignment branch
See issue: https://github.com/jnothman/UpSetPlot/issues/303
'''

import yaml
from pathlib import Path
import pandas as pd
from upsetplot import from_contents
from upsetplot import UpSet

panel_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/panels/hvg-100/cells')
hvgs = {}
for panel_path in panel_dir.glob('*.yaml'):
    panel = yaml.safe_load(panel_path.open())
    hvgs[panel_path.stem] = panel['target_panel']

hvgs = from_contents(hvgs)
axs = UpSet(hvgs).plot()
axs['matrix'].figure.show()
axs['matrix'].figure.savefig(panel_dir / 'upset.png')