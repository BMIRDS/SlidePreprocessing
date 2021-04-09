import argparse
import yaml
from pathlib import Path

""" Config file parser
"""

#TODO: support logging
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/config.yaml')
    # parser.add_argument('--log_dir', '-l', type=str, default='log')
    args = parser.parse_args()
    config_file = args.config

    print(f"[info] Loading config file from: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    class Config:
        """ Convert a config dictionary to config object
        """
        def __init__(self, **entries):
            self.__dict__.update(entries)

    configs = Config(**config)
    # Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    return configs
