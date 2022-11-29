from definitions import BirdClassifierArguments

from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_config():
    cli_config = OmegaConf.from_cli()
    if "config" not in cli_config:
        raise ValueError(
                "Please pass the 'config' to specify configuration yaml")
    yaml_conf = OmegaConf.load(cli_config.config)
    conf = instantiate(yaml_conf)
    cli_config.pop("config")
    config : BirdClassifierArguments = OmegaConf.merge(conf, cli_config)
    return config