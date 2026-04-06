from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

from xenium_hne_fusion.train.config import Config
from xenium_hne_fusion.train.supervised import main, train

assert load_dotenv(override=True)

# Manual debug entrypoint for quick local iteration.
# Uncomment to run without going through the CLI.
# fast_dev_run = debug = True
# cfg = Config.from_yaml(Path("configs/train/beat/expression/early-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/late-fusion.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/vision.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-token.yaml"))
# cfg = Config.from_yaml(Path("configs/train/beat/expression/expr-tile.yaml"))
# train(cfg, debug=debug)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
