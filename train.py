import numpy as np
import tensorflow as tf

import sys
import logging

from utils.config import load_config, get_data_config, get_training_config

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
    data_config = get_data_config(config)
    training_config = get_training_config(config)

    subject_names = data_config["subjects"]
    sequence_names = data_config["sequences"]

    data_handler = DataHandler()

    train_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["train"],
        sequence_names=sequence_names["train"],
        batch_size=data_config["batch_size"],
        window_size=2,
    )

    val_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["val"],
        sequence_names=sequence_names["val"],
        batch_size=data_config["batch_size"],
        window_size=2,
    )
    test_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["test"],
        sequence_names=sequence_names["test"],
        batch_size=data_config["batch_size"],
        window_size=2,
    )

    model = Model(
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        test_batcher=test_batcher,
        learning_rate=training_config["learning_rate"],
        epochs=training_config["epochs"],
        validation_steps=training_config["validation_steps"],
    )

    model.train()
    # model.eval()
    model.save(dir_path=config["model_dir"])


if __name__ == "__main__":
    main()
