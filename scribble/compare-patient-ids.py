import pandas as pd
from pathlib import Path

# pre_xe_ids = {'1H4D', '1HLN', '1I31', '1FYP', '1IVQ', '1J31', '1G4J', '1H20', '1HYU', '1GV3', '1IM1', '1G4X', '1GPT', '1GNX', '1G11', '1GXI', '1HJC', '1GJA', '1GNV', '1IVR', '1HS7', '1HRW', '1H5K', '1IJZ', '1FZX', '1GPZ', '1IZX', '1HHP', '1IR5', '1G4Y', '1IUK', '1HUJ', '1H5M', '1H5O', '1IKT', '1GY3', '1GCY', '1HSB', '1IQK', '1IS3', '1GY0', '1G9B', '1GPO', '1FYV', '1J4I', '1GY1', '1GUG', '1GK0', '1IWV', '1G97', '1GEM', '1GQ8', '1H49', '1HJ3', '1GCZ', '1ISE', '1J1D', '1GX4', '1IS2', '1IPZ', '1GGF', '1J7Y', '1FY2', '1GVX', '1JAS', '1GGR', '1HS6', '1I33', '1JAT', '1FYN', '1HJ4', '1J2Y', '1IUV', '1HKD', '1HLW', '1FU3', '1J4J', '1J4B', '1G4K', '1HJD', '1HJ7', '1J2X', '1GS0', '1H4E', '1HJ6', '1HQR', '1G9J', '1G52', '1IZY', '1FU2', '1FYO', '1IO9', '1GS6', '1H2J', '1GB9', '1IRS', '1H5I', '1GRQ', '1HFS', '1GBF', '1I30', '1G3X', '1I32', '1G3T', '1HQA', '1G12', '1H0T', '1G3Y', '1HUX', '1GJB', '1GTE', '1J30', '1G5B', '1H0U', '1J00', '1J25', '1H5L', '1H1Q', '1G4Z', '1IXC', '1GP1', '1HKH', '1GPD', '1IK0', '1HRV', '1HLM', '1H4A', '1GP0', '1HLL', '1FZW', '1G51', '1G96', '1GEG', '1HJ5', '1GJC', '1JAR', '1GEP', '1IM0', '1GPP', '1FYT', '1FXS', '1HHR', '1HKG', '1IRG', '1FPK', '1IQJ', '1FP2', '1G3U', '1FYS', '1G6Z'}
# len(pre_xe_ids)

# base_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/02_processed/beat')
# df = pd.read_parquet(base_dir / 'clinical.parquet')
# df.set_index('sample_id').clinical_pat_id.to_dict()

sid_to_pid = {'XE_1CGZ.01_42_HNE_1CGZ': '1GCZ', 'XE_1FP2.01_HNE_1FP2-01': '1FP2', 'XE_1FPK.01_HNE_1FPK-01': '1FPK',
              'XE_1FU2.01_HNE_1FU2': '1FU2', 'XE_1FU3.01_HNE_1FU3-01': '1FU3', 'XE_1FXS.01_HNE_1FXS01': '1FXS',
              'XE_1FY2.01_HNE_1FY2': '1FY2', 'XE_1FYN.01_HNE_1FYN-01': '1FYN', 'XE_1FYO.01_HNE_1FYO01': '1FYO',
              'XE_1FYP.01_HNE_1FYP': '1FYP', 'XE_1FYS.01_HNE_1FYS': '1FYS', 'XE_1FYT.01_HNE_1FYT': '1FYT',
              'XE_1FYV.01_HNE_1FYV-01': '1FYV', 'XE_1FZW.01_HNE_1FZW': '1FZW', 'XE_1FZX.01_HNE_1FZX01': '1FZX',
              'XE_1G11.01_HNE_1G11': '1G11', 'XE_1G11.02_HNE_1G11': '1G11', 'XE_1G11.03_HNE_1G1103': '1G11',
              'XE_1G12.01_HNE_1G12': '1G12', 'XE_1G3S.01_HNE_1G3S-01': '1G3S', 'XE_1G3U.01_HNE_1G3U': '1G3U',
              'XE_1G3X.01_HNE_1G3X-01': '1G3X', 'XE_1G3Y.01_HNE_1G3Y': '1G3Y', 'XE_1G40.01_HNE_1G40-01': '1G40',
              'XE_1G4J.01_HNE_1G4J-01': '1G4J', 'XE_1G4K.01_HNE_1G4K': '1G4K', 'XE_1G4X.01_42_HNE_1G4X': '1G4X',
              'XE_1G4Y.01_HNE_1G4Y': '1GAK', 'XE_1G4Z.01_HNE_1G4Z': '1G4Z', 'XE_1G50.01_HNE_1G50': '1G50',
              'XE_1G51.01_HNE_1G51': '1G51', 'XE_1G52.01_24_HNE_1G52-01': '1G52', 'XE_1G5A.01_HNE_1G5A': '1G5A',
              'XE_1G5A.02_HNE_1G5A': '1G5A', 'XE_1G5B.01_HNE_1G5B-01': '1G5B', 'XE_1G63.01_HNE_1G63-01': '1G63',
              'XE_1G6Z.01_HNE_1G6Z-01': '1G6Z', 'XE_1G96.01_HNE_1G96': '1G96', 'XE_1G97.01_HNE_1G97-01': '1G97',
              'XE_1G9C.01_HNE_1G9C': '1G9C', 'XE_1G9J.01_24_HNE_1G9J-01': '1G9J', 'XE_1GB9.01_HNE_1GB9': '1GB9',
              'XE_1GBF.01_HNE_1GBF-01': '1GBF', 'XE_1GCY.01_HNE_1GCY': '1GCY', 'XE_1GEG.01_HNE_1GEG': '1GEG',
              'XE_1GEM.01_HNE_1GEM-01': '1GEM', 'XE_1GEP.01_HNE_1GEP': '1GEP', 'XE_1GGF.01_HNE_1GGF': '1GGF',
              'XE_1GGR.01_HNE_1GGR': '1GGR', 'XE_1GJA.01_24_HNE_1GJA-01': '1GJA', 'XE_1GJB.01_HNE_1GJB': '1GJB',
              'XE_1GJC.01_HNE_1GJC': '1GJC', 'XE_1GK0.01_HNE_1GK0': '1GK0', 'XE_1GNX.01_HNE_1GNX': '1GNX',
              'XE_1GP0.01_HNE_1GPO': '1GP0', 'XE_1GP1.01_HNE_1GP1': '1GP1', 'XE_1GPD.01_HNE_1GPD': '1GPD',
              'XE_1GPO.01_HNE_1GPO': '1GPO', 'XE_1GPP.01_HNE_1GPP-01': '1GPP', 'XE_1GPT.01_HNE_1GPT': '1GPT',
              'XE_1GPZ.01_HNE_1GPZ': '1GPZ', 'XE_1GQ8.01_HNE_1GQ8': '1GQ8', 'XE_1GQR.01_HNE_1GQR': '1GRQ',
              'XE_1GS0.01_HNE_1GS0-01': '1GS0', 'XE_1GS6.01_HNE_1GS6': '1GS6', 'XE_1GTE.01_HNE_1GTE-01': '1GTE',
              'XE_1GTK.01_HNE_1GTK-01': '1GTK', 'XE_1GUG.01_HNE_1GUG': '1GUG', 'XE_1GV3.01_HNE_1GV3': '1GV3',
              'XE_1GVN.01_HNE_1GVN': '1GNV', 'XE_1GVX.01_HNE_1GVX-01': '1GVX', 'XE_1GWV.01_HNE_1GWV': '1GWV',
              'XE_1GX4.01_HNE_1GX4-01': '1GX4', 'XE_1GXI.01_HNE_1GXI-01': '1GXI', 'XE_1GY0.01_HNE_1GY0': '1GY0',
              'XE_1GY1.01_HNE_1GY1-01': '1GY1', 'XE_1GY3.01_HNE_1GY3-01': '1GY3', 'XE_1H0S.01_HNE_1H0S-01': '1H0S',
              'XE_1H0T.01_HNE_1H0T': '1H0T', 'XE_1H0U.01_24_HNE_1H0U-01': '1H0U', 'XE_1H20.01_HNE_1H20_01': '1H20',
              'XE_1H3W.01_HNE_1H3W_01': '1H3W', 'XE_1H49.01_HNE_1H49_01_-_5': '1H49',
              'XE_1H49.03_HNE_1H49_03_-_7': '1H49', 'XE_1H49.04_HNE_1H49-04': '1H49', 'XE_1H4A.01_HNE_1H4A': '1H4A',
              'XE_1H4D.01_HNE_1H4D': '1H4D', 'XE_1H4E.01_HNE_1H4E-01': '1H4E', 'XE_1H50.01_HNE_1H50-01': '1H5O',
              'XE_1H5I.01_HNE_1H5I': '1H5I', 'XE_1H5K.01_27_HNE_1H5K_01': '1H5K', 'XE_1H5M.01_HNE_1H5M-01': '1H5M',
              'XE_1HDD.01_HNE_1HDD': '1HHD', 'XE_1HFS.01_HNE_1HFS': '1HFS', 'XE_1HHP.0B_HNE_1HHP-0B': '1HHP',
              'XE_1HHR.01_HNE_1HHR-01': '1HHR', 'XE_1HJ3.01_HNE_1HJ3-01': '1HJ3', 'XE_1HJ4.01_42_HNE_HJ4': '1HJ4',
              'XE_1HJ6.01_HNE_1HJ6-01': '1HJ6', 'XE_1HJ7.01_HNE_1HJ7': '1HJ7', 'XE_1HJC.01_HNE_1HJC': '1HJC',
              'XE_1HJD.01_HNE_1HJD': '1HJD', 'XE_1HKD.01_HNE_1HKD-01': '1HKD', 'XE_1HKG.01_HNE_1HKG-01': '1HKG',
              'XE_1HKH.01_HNE_1HKH_01': '1HKH', 'XE_1HL5.01_HNE_1HL5_01': '1H5L', 'XE_1HLM.01_HNE_1HLM': '1HLM',
              'XE_1HLN.01_HNE_1HLN': '1HLN', 'XE_1HLW.01_HNE_1HLW': '1HLW', 'XE_1HQA.01_HNE_1HQA-01': '1HQA',
              'XE_1HQR.01_HNE_1HQR-01': '1HQR', 'XE_1HRW.01_HNE_1HRW_01': '1HRW', 'XE_1HS6.01_HNE_1Hs6': '1HS6',
              'XE_1HS7.01_HNE_1HS7': '1HS7', 'XE_1HUJ.01_HNE_1HUJ-01': '1HUJ', 'XE_1HUX.01_HNE_1HUX': '1HUX',
              'XE_1HV3.01_HNE_1HV3': '1HV3', 'XE_1HYU.01_HNE_1HYU_01': '1HYU', 'XE_1I30.01_HNE_1I30-01': '1I30',
              'XE_1I31.01_HNE_1I31': '1I31', 'XE_1I32.01_HNE_1I32': '1I32', 'XE_1I33.01_HNE_1I33_01': '1I33',
              'XE_1IJZ.01_HNE_1IJZ': '1IJZ', 'XE_1IK0.01_HNE_1IK0': '1IK0', 'XE_1IKT.01_HNE_1IKT': '1IKT',
              'XE_1IM0.01_HNE_1IM0-01': '1IM0', 'XE_1IM0.0C_HNE_1IM0-0C': '1IM0', 'XE_1IO9.01_HNE_1IO9': '1IO9',
              'XE_1IPZ.01_HNE_1IPZ': '1IPZ', 'XE_1IQJ.01_HNE_1IQJ-01': '1IQJ', 'XE_1IQK.01_HNE_1IQK': '1IQK',
              'XE_1IR5.01_HNE_1IR5': '1IR5', 'XE_1IRS.01_HNE_1IRS-01': '1IRS', 'XE_1IS2.05_HNE_1IS2-05': '1IS2',
              'XE_1IS2.06_HNE_1IS2-06': '1IS2', 'XE_1IS3.01_HNE_1IS3': '1IS3', 'XE_1IUK.01_HNE_1IUK': '1IUK',
              'XE_1IUV.01_HNE_1IUV': '1IUV', 'XE_1IUW.01_HNE_1IUW': '1IUW', 'XE_1IVQ.01_HNE_1IVQ_01_-_4': '1IVQ',
              'XE_1IVQ.02_HNE_1IVQ': '1IVQ', 'XE_1IVR.01_HNE_1IVR_01-1': '1IVR', 'XE_1IVR.02_HNE_1IVR_02_-_2': '1IVR',
              'XE_1IVR.03_HNE_1IVR': '1IVR', 'XE_1IVR.04_HNE_1IVR_04_-_3': '1IVR', 'XE_1IWV.01_HNE_1IWV-01': '1IWW',
              'XE_1IXC.05_HNE_1IXC-05': '1IXC', 'XE_1IZX.01_HNE_1IZX': '1IZX', 'XE_1IZY.01_HNE_1IZY': '1IZY',
              'XE_1J25.01_HNE_1J25': '1J25', 'XE_1J2X.01_HNE_1J2X': '1J2X', 'XE_1J2Y.01_HNE_1J2Y': '1J2Y',
              'XE_1J30.01_HNE_1J30': '1J30', 'XE_1J31.01_HNE_1J31': '1J31', 'XE_1J4B.01_HNE_1J4B': '1J4B',
              'XE_1J4I.01_HNE_1J4I': '1J4I', 'XE_1J4J.01_HNE_1J4J': '1J4J', 'XE_1J7Y.01_HNE_1J7Y': '1J7Y',
              'XE_1JAR.01_HNE_1JAR': '1JAR', 'XE_1JAS.01_HNE_1JAS': '1JAS', 'XE_1JAT.01_HNE_1JAT': '1JAT',
              'XE_1JID.01_HNE_1JID': '1J1D', 'XE_1JOO.01_HNE_1JOO': '1J00'}

# post_xe_ids = set(df.clinical_pat_id)
# split pre_xe_ids stratified on histology

# %% create splits
items = pd.read_json(
    '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v0/03_output/beat/items/cells.json')
items['clinical_pat_id'] = items.sample_id.map(sid_to_pid)
assert not items['clinical_pat_id'].isna().any()

sample_ids = set(items.sample_id)
assert len(sample_ids) == 149

clinical_pat_ids = set(items.clinical_pat_id)
assert len(clinical_pat_ids) == 139

meta_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/metadata_b.parquet')
meta = pd.read_parquet(meta_path)
'7b' in meta.columns

# meta = meta[meta.sample_id.isin(sample_ids)]
meta = meta[meta.clinical_pat_id.isin(clinical_pat_ids)]
meta = meta.drop_duplicates()
assert len(meta) == 139

from sklearn.model_selection import RepeatedStratifiedKFold
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
splits = []
for i, (train, test) in enumerate(splitter.split(meta, y=meta['7b'])):
    split = {
        'id': i,
        'train_ids': meta.iloc[train].clinical_pat_id.tolist(),
        'test_ids': meta.iloc[test].clinical_pat_id.tolist(),
    }
    splits.append(split)

for i in range(len(splits)):
    assert set(splits[i]['test_ids']).intersection(set(splits[i]['train_ids'])) == set()

import json
save_path = meta_path.parent / f'{meta_path.stem}_splits.json'
save_path.write_text(json.dumps(splits))
