#!/usr/bin/env bash

set -euo pipefail

python - <<'PY'
from __future__ import annotations

from importlib import metadata


def get_version(module: object, dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", "unknown")


def main() -> None:
    import ai4bmr_learn
    from ai4bmr_learn import data

    import lightning
    import torch

    import xenium_hne_fusion
    from xenium_hne_fusion.train.config import Config
    from xenium_hne_fusion.train.supervised import main, train

    modules = [
        ("xenium_hne_fusion", xenium_hne_fusion, "xenium-hne-fusion"),
        ("ai4bmr_learn", ai4bmr_learn, "ai4bmr-learn"),
        ("torch", torch, "torch"),
        ("lightning", lightning, "lightning"),
    ]

    for name, module, dist_name in modules:
        module_path = getattr(module, "__file__", "<namespace>")
        version = get_version(module, dist_name)
        print(f"{name}: version={version} path={module_path}")


if __name__ == "__main__":
    main()
PY
