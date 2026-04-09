from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.train.cli import build_train_parser, namespace_to_config
from xenium_hne_fusion.train.config import (
    Config,
)
from xenium_hne_fusion.train.supervised import main, train

load_dotenv(override=True)

# Manual debug entrypoint for quick local iteration.
# Uncomment to run without going through the CLI.
# fast_dev_run = debug = True
# cfg = Config.from_yaml(Path("configs/train/beat/expression/early-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/late-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/vision.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-token.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-tile.yaml"))
# train(cfg, debug=debug)


def _build_parser():
    return build_train_parser()


def _namespace_to_config(ns) -> Config:
    return namespace_to_config(ns)


if __name__ == "__main__":
    parser = _build_parser()
    namespace = parser.parse_args()
    main(_namespace_to_config(namespace))
