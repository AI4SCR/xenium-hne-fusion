from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from xenium_hne_fusion.train.mil import train
from xenium_hne_fusion.train.mil_config import MILConfig

load_dotenv(override=True)

CONFIG_PATH = Path("configs/mil/beat/classification.yaml")

cfg = MILConfig.from_yaml(CONFIG_PATH)
cfg.debug = True
train(cfg, debug=cfg.debug)
