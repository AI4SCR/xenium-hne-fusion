from xenium_hne_fusion.train.config import TrainingConfig
from xenium_hne_fusion.train.supervised import main, train

# Manual debug entrypoint for quick local iteration.
# Uncomment to run without going through the CLI.
# from dotenv import load_dotenv
# load_dotenv(override=True)
# from pathlib import Path
# debug = True
# cfg = TrainingConfig.from_yaml(Path("configs/train/owkin/proteins/early-fusion.yaml"))
# train(cfg, debug=debug)


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config", required=True)
    parser.add_class_arguments(TrainingConfig, nested_key="train")
    parser.add_argument("--debug", type=bool, default=False)
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    config_path = cfg.as_dict().get("config")
    init = parser.instantiate(cfg)
    main(init.train, debug=init.debug, config_path=config_path)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
