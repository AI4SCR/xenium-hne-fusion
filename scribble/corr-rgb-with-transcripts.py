from dotenv import load_dotenv
assert load_dotenv(override=True)

from loguru import logger
from torch.utils.data import DataLoader

from xenium_hne_fusion.datasets.tiles import TileDataset
from xenium_hne_fusion.utils.getters import get_managed_paths
import yaml

def main() -> None:

    save_dir = get_managed_paths('beat').output_dir / 'results'
    logger.info(f'Saving results to {save_dir}')

    items_path = get_managed_paths('beat').output_dir / 'items' / 'cells.json'
    panel_path = get_managed_paths('beat').output_dir / 'panels' / 'default.yaml'
    source_panel = yaml.load(panel_path.open(), Loader=yaml.SafeLoader)['source_panel']
    ds = TileDataset(items_path=items_path, target='rgb',
                     source_panel=source_panel,
                     include_expr=True, include_image=True)
    ds.setup()
    item = ds[0]
    item['modalities']['expr_tokens'].sum(dim=(-2))
    item['modalities']['image']
    item['rgb']

    batch_size, num_workers = 32, 8
    dataloader_kws = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )

    dl = DataLoader(ds, **dataloader_kws)
    batch = next(iter(dl))

    targets = []
    exprs = []
    import torch
    from tqdm import tqdm
    i = 0
    for batch in tqdm(dl):
        expr = batch['modalities']['expr_tokens'].sum(dim=(-2))
        target = batch['rgb']

        targets.append(target)
        exprs.append(expr)

        i += 1
        if i == 5000:
            break

    exprs = torch.concat(exprs)
    targets = torch.concat(targets)

    expr_sum = exprs.sum(dim=1)
    expr_sum = torch.log1p(expr_sum)

    import seaborn as sns
    import matplotlib.pyplot as plt
    for i, y in enumerate('rgb'):
        ax = sns.scatterplot(x=expr_sum, y=targets[:, i], s=1, linewidth=0)
        ax.set_xlabel('expr_sum')
        ax.set_ylabel(y)
        ax.figure.show()
        ax.figure.tight_layout()
        ax.figure.savefig(save_dir / f'{y}.png')
        plt.close('all')

main()