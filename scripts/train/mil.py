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
    init = parser.instantiate_classes(cfg)
    d = vars(init)
    d.pop("config", None)

    raise SystemExit(main(MILConfig(**d)))