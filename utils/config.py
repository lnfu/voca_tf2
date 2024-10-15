import os
import logging
import yaml
import argparse


def load_config_from_file(config_file_path: str) -> dict:
    if not os.path.exists(config_file_path):
        logging.error(f"Configuration file {config_file_path} does not exist.")
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")

    try:
        with open(config_file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {config_file_path}. Error: {e}")
        raise


def load_config(config_file_path: str) -> dict:
    config = load_config_from_file(config_file_path)

    flags = parse_cli_flags()
    config["training"]["batch_size"] = flags.batch_size
    config["training"]["learning_rate"] = flags.learning_rate
    config["training"]["epochs"] = flags.epochs
    config["training"]["validation_freq"] = flags.validation_freq
    return config


def get_data_config(config: dict) -> dict:
    return config.get("data", {})


def get_training_config(config: dict) -> dict:
    return config.get("training", {})


def parse_cli_flags():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--validation_freq", type=int, default=1, help="Frequency of validation during training")

    return parser.parse_args()
