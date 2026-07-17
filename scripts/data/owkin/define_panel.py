from pathlib import Path
import pandas as pd
import yaml

from xenium_hne_fusion.processing import load_feature_universe
from xenium_hne_fusion.utils.getters import ManagedPaths

paths = ManagedPaths(
    data_dir=Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v4'),
    name='owkin',
)

sample_ids = sorted(p.name for p in paths.processed_dir.iterdir() if p.is_dir())

groups = {}
for sample_id in sample_ids:
    _, group, *_ = sample_id.split('_')
    groups.setdefault(group, []).append(sample_id)

panels = {}
for group, group_sample_ids in sorted(groups.items()):
    gene_universes = [
        set(load_feature_universe(paths.processed_dir / sample_id / 'feature_universe.txt'))
        for sample_id in group_sample_ids
    ]
    for sample_id, universe in zip(group_sample_ids, gene_universes):
        assert universe == gene_universes[0], f'Feature universe mismatch in group {group} for sample {sample_id}'

    protein_universes = [
        set(pd.read_parquet(paths.structured_dir / sample_id / 'proteins.parquet').columns) - {'geometry'}
        for sample_id in group_sample_ids
    ]
    for sample_id, universe in zip(group_sample_ids, protein_universes):
        assert universe == protein_universes[0], f'Protein universe mismatch in group {group} for sample {sample_id}'

    panels[group] = {
        'genes': sorted(gene_universes[0]),
        'proteins': sorted(protein_universes[0]),
    }

save_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/panels/owkin/owkin.yaml')
save_path.parent.mkdir(parents=True, exist_ok=True)
save_path.write_text(yaml.safe_dump(panels, sort_keys=True))
