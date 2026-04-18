import sys

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.train.config import (
    BackboneConfig,
    Config,
    DataLoaderConfig,
    HeadConfig,
    LitConfig,
    TaskConfig,
    TrainerConfig,
    WandbConfig,
)
from xenium_hne_fusion.train.supervised import main, train

load_dotenv(override=True)

# Manual debug entrypoint for quick local iteration.
# Uncomment to run without going through the CLI.
# fast_dev_run = debug = True
# from pathlib import Path
# cfg = Config.from_yaml(Path("configs/train/beat/expression/early-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/late-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/vision.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-token.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-tile.yaml"))
# train(cfg, debug=debug)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_class_arguments(TaskConfig, nested_key="task")
    parser.add_class_arguments(HeadConfig, nested_key="head")
    parser.add_class_arguments(BackboneConfig, nested_key="backbone")
    parser.add_class_arguments(DataLoaderConfig, nested_key="data")
    parser.add_class_arguments(LitConfig, nested_key="lit")
    parser.add_class_arguments(TrainerConfig, nested_key="trainer")
    parser.add_class_arguments(WandbConfig, nested_key="wandb")
    return parser


def _namespace_to_config(ns) -> Config:
    data = ns.as_dict()
    return Config(
        debug=data["debug"],
        task=TaskConfig(**data["task"]),
        head=HeadConfig(**data["head"]),
        backbone=BackboneConfig(**data["backbone"]),
        data=DataLoaderConfig(**data["data"]),
        lit=LitConfig(**data["lit"]),
        trainer=TrainerConfig(**data["trainer"]),
        wandb=WandbConfig(**data["wandb"]),
    )


def cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    config_path = namespace.as_dict().get("config")
    main(_namespace_to_config(namespace), config_path=config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
