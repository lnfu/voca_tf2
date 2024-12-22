import logging
import yaml

from .common import check_file_exists


def load_config(config_file_path: str) -> dict:
    check_file_exists(config_file_path)
    try:
        with open(config_file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error(
            f"Error parsing YAML file: {config_file_path}. Error: {e}")
        raise
