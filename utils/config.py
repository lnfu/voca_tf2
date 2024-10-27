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
    config = overwrite_config_from_cli(config)
    return config


def get_data_config(config: dict) -> dict:
    return config.get("data", {})


def get_training_config(config: dict) -> dict:
    return config.get("training", {})


def overwrite_config_from_cli(config: dict):
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["training"]["learning_rate"],
        help="Learning rate for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["training"]["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"]["epochs"],
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=config["training"]["validation_freq"],
        help="Frequency of validation during training",
    )

    args = parser.parse_args()

    config["training"].update(
        {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "validation_freq": args.validation_freq,
        }
    )

    return config
