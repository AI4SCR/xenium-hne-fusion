import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

import pandas as pd
import torch
from jsonargparse import ArgumentParser
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field

import lazyslide as zs
from lazyslide.models import MODEL_REGISTRY

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.utils.getters import get_managed_paths

@dataclass
class FMDataConfig:
    name: str
    master_path: Path
    items_path: Path
    batch_size: int = 256


@dataclass
class FMModelConfig:
    name: str = 'conch_v1.5'


@dataclass
class FMEmbeddingsConfig:
    data: FMDataConfig
    model: FMModelConfig = field(default_factory=FMModelConfig)
    overwrite: bool = False

import yaml
cfg = yaml.load(Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/configs/artifacts/beat/unil/fm.yaml').open(), Loader=yaml.SafeLoader)
cfg = FMEmbeddingsConfig(
    data=FMDataConfig(**cfg['data']),
    model=FMModelConfig(**cfg['model'])
)
cfg.model.name = 'uni'
cfg.data.items_path = Path(cfg.data.items_path)

# def main(cfg: FMEmbeddingsConfig) -> int:
def main() -> int:
    import yaml
    cfg = yaml.load(Path(
        '/work/FAC/FBM/DBC/mrapsoma/prometex/projects/xenium-hne-fusion/configs/artifacts/beat/unil/fm.yaml').open(),
                    Loader=yaml.SafeLoader)
    cfg = FMEmbeddingsConfig(
        data=FMDataConfig(**cfg['data']),
        model=FMModelConfig(**cfg['model'])
    )
    cfg.model.name = 'uni'
    cfg.data.items_path = Path(cfg.data.items_path)
    cfg.data.batch_size = 2

    logger.info(f'Creating embeddings with {cfg.model.name}')

    paths = get_managed_paths(cfg.data.name)
    output_dir = paths.output_dir

    items_path = cfg.data.items_path
    if not items_path.is_absolute():
        items_path = output_dir / 'items' / items_path
    assert items_path.exists(), f'items_path not found: {items_path}'

    ds = TileDataset(items_path=items_path, target='cell_types', include_expr=False)
    ds.setup()
    assert ds[0]['modalities']['image'].dtype == torch.uint8

    sample_ids = {i['sample_id'] for i in ds.items}

    master = pd.read_csv(cfg.data.master_path)
    include = master.paired_id.isin(sample_ids) & (master.tech_id == 'XE')
    sid_to_pid = master.loc[include].set_index('paired_id')['clinical_pat_id'].to_dict()
    assert set(sid_to_pid) == sample_ids, f'master coverage mismatch: {sample_ids - set(sid_to_pid)}'

    model_name = cfg.model.name
    assert model_name.lower() in zs.models.list_models(), f'Unknown model: {model_name}'
    # zs.tl.feature_extraction()
    # model_name = 'conch_v1.5'

    # torch.set_default_device("cpu")  # required to avoid: RuntimeError: Tensor.item() cannot be called on meta tensors
    model = MODEL_REGISTRY[model_name]()
    transform = model.get_transform()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.model.eval()

    save_dir = output_dir / 'fm_embeddings' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    items = list(ds.items)
    logger.info(f'Processing {len(sample_ids)} samples with {model_name}.')

    for sid in sample_ids:
        out_path = save_dir / f'{sid}.parquet'
        if out_path.exists() and not cfg.overwrite:
            logger.info(f'Skip {sid} (already exists).')
            continue

        logger.info(f'Computing {model_name} embeddings for {sid}.')
        ds.items = [i for i in items if i['sample_id'] == sid]
        dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=10)

        records = []
        for batch in tqdm(dl, desc=sid):
            images = batch.pop('modalities')['image'].to(device)
            z = model(transform(images))

            del batch['target']
            index = pd.DataFrame(batch)
            index['clinical_pat_id'] = index.sample_id.map(sid_to_pid)
            index = index.convert_dtypes().astype('category')
            index = pd.MultiIndex.from_frame(index)

            records.append(pd.DataFrame(z.cpu().numpy(), index=index))

        result = pd.concat(records)
        assert result.index.is_unique
        result.to_parquet(out_path)

    return 0

main()

#%%
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config", action="config")
#     parser.add_class_arguments(FMEmbeddingsConfig, None)
#
#     cfg = parser.parse_args()
#     init = parser.instantiate_classes(cfg)
#     d = vars(init)
#     d.pop("config", None)
#     raise SystemExit(main(FMEmbeddingsConfig(**d)))
