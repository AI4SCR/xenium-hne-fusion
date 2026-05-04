"""Test what instantiate_classes returns for MILConfig parsed via jsonargparse.

Run:
    uv run python scribble/test_jsonargparse_instantiate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jsonargparse import ArgumentParser
from xenium_hne_fusion.train.mil_config import MILConfig

parser = ArgumentParser()
parser.add_argument("--config", action="config")
parser.add_class_arguments(MILConfig, None)

cfg = parser.parse_args([])          # no args — use defaults
init = parser.instantiate_classes(cfg)

print(f"type(init)            : {type(init)}")
print(f"isinstance(Namespace) : {type(init).__name__}")
print(f"init                  :\n{init}")

# Check whether we get a real MILConfig dataclass or a Namespace
is_dataclass_instance = isinstance(init, MILConfig)
print(f"\nis MILConfig dataclass: {is_dataclass_instance}")

if not is_dataclass_instance:
    print("\nFields accessible via attribute access:")
    for k in vars(init):
        print(f"  {k}: {getattr(init, k)!r}")
