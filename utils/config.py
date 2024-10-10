import os
import logging
import yaml


def load_config(config_file_path) -> dict:
    if not os.path.exists(config_file_path):
        logging.error(f"Configuration file {config_file_path} does not exist.")
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")

    try:
        with open(config_file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {config_file_path}. Error: {e}")
        raise


def get_data_config(config: dict) -> dict:
    return config.get("data", {})


def get_training_config(config: dict) -> dict:
    return config.get("training", {})
