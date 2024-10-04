import tensorflow as tf

import sys
import logging

from utils.config import load_config

from utils.data_handlers.data_handler import DataHandler
from utils.batcher import Batcher
from utils.model import Model

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    data_config = config["data"]

    subject_names = data_config["subjects"]
    sequence_names = data_config["sequences"]

    data_handler = DataHandler(
        subject_names=subject_names, sequence_names=sequence_names
    )

    batcher = Batcher(data_handler=data_handler, batch_size=data_config["batch_size"])

    training_config = config["training"]
    model = Model(
        batcher=batcher,
        learning_rate=training_config["learning_rate"],
        epochs=training_config["epochs"],
        validation_steps=training_config["validation_steps"],
    )

    model.train()


if __name__ == "__main__":
    main()
