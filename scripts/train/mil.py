from dotenv import load_dotenv

load_dotenv(override=True)

from xenium_hne_fusion.train.mil import main
from xenium_hne_fusion.train.mil_config import MILConfig


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(MILConfig, None)

    cfg = parser.parse_args()
    config_path = cfg.as_dict().get("config")
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)

    raise SystemExit(main(MILConfig(**d), config_path=config_path))