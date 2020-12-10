import argparse
import yaml

""" Config file parser
"""

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/config.yaml')
    args = parser.parse_args()
    config_file = args.config

    print(f"Loading config file from: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    class Config:
        """ Convert a config dictionary to config object
        """
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Config(**config)
