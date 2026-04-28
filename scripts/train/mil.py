import sys

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from xenium_hne_fusion.train.config import HeadConfig, TrainerConfig, WandbConfig
from xenium_hne_fusion.train.mil import main
from xenium_hne_fusion.train.mil_config import (
    AggregatorConfig,
    MILConfig,
    MILDataConfig,
    MILLitConfig,
    MILTaskConfig,
    PretrainedConfig,
)

load_dotenv(override=True)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_class_arguments(PretrainedConfig, nested_key="pretrained")
    parser.add_class_arguments(MILDataConfig, nested_key="data")
    parser.add_class_arguments(MILTaskConfig, nested_key="task")
    parser.add_class_arguments(AggregatorConfig, nested_key="aggregator")
    parser.add_class_arguments(HeadConfig, nested_key="head")
    parser.add_class_arguments(MILLitConfig, nested_key="lit")
    parser.add_class_arguments(TrainerConfig, nested_key="trainer")
    parser.add_class_arguments(WandbConfig, nested_key="wandb")
    return parser


def _namespace_to_config(ns) -> MILConfig:
    data = ns.as_dict()
    return MILConfig(
        debug=data["debug"],
        pretrained=PretrainedConfig(**data["pretrained"]),
        data=MILDataConfig(**data["data"]),
        task=MILTaskConfig(**data["task"]),
        aggregator=AggregatorConfig(**data["aggregator"]),
        head=HeadConfig(**data["head"]),
        lit=MILLitConfig(**data["lit"]),
        trainer=TrainerConfig(**data["trainer"]),
        wandb=WandbConfig(**data["wandb"]),
    )


def cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    main(_namespace_to_config(namespace), debug=namespace.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
