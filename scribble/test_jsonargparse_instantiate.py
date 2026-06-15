from jsonargparse import ArgumentParser
from pydantic import BaseModel

class Config(BaseModel):
    a: str = 'a'
    b: int = 1

parser = ArgumentParser()
parser.add_argument("--config", action="config")
# parser.add_class_arguments(Config, None)  # add class parameters
parser.add_class_arguments(Config, 'myconfig')  # add class parameters

cfg = parser.parse_args([])
cfg = parser.instantiate(cfg)
isinstance(cfg.myconfig, Config)