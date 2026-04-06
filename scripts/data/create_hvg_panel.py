"""Generate a tile-level Scanpy HVG panel from fit-split tiles."""

from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli
from loguru import logger

load_dotenv()

from xenium_hne_fusion.hvg import create_hvg_panel, load_hvg_recipe
from xenium_hne_fusion.metadata import get_default_split_path, load_split_config
from xenium_hne_fusion.utils.getters import load_pipeline_config


def main(
    dataset: str,
    recipe_path: Path | None = None,
    config_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    cfg = load_pipeline_config(dataset, config_path)
    recipe_path = recipe_path or (Path('configs/panel_recipes') / dataset / 'default.yaml')
    recipe = load_hvg_recipe(recipe_path)
    split_cfg = load_split_config(Path('configs/splits') / f'{dataset}.yaml')

    items_path = cfg.output_dir / 'items' / f'{recipe.items_name}.json'
    split_dir = cfg.output_dir / 'splits' / recipe.split_name
    split_metadata_path = get_default_split_path(split_dir, split_cfg)
    output_path = cfg.output_dir / 'panels' / f'{recipe.panel_name}.yaml'

    if output_path.exists() and not overwrite:
        logger.info(f'HVG panel already exists: {output_path}')
        return

    create_hvg_panel(
        items_path=items_path,
        split_metadata_path=split_metadata_path,
        output_path=output_path,
        n_top_genes=recipe.n_top_genes,
        flavor=recipe.flavor,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    auto_cli(main)
