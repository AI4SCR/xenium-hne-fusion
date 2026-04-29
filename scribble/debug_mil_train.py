from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.train.mil import train
from xenium_hne_fusion.train.mil_config import MILConfig

load_dotenv(override=True)

CONFIG_PATH = Path("<PATH>")

def main() -> None:
    cfg = MILConfig.from_yaml(CONFIG_PATH)
    train(cfg, debug=cfg.debug)


if __name__ == "__main__":
    main()
