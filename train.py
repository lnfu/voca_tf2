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

    data_handler = DataHandler(
        audio_raw_path=data_config["path"]["audio"]["raw"],
        audio_processed_path=data_config["path"]["audio"]["processed"],
        pcd_data_path=data_config["path"]["pcd"]["data"],
        pcd_index_path=data_config["path"]["pcd"]["index"],
        pcd_template_path=data_config["path"]["pcd"]["template"],
    )

    subject_names = data_config["subjects"]
    sequence_names = data_config["sequences"]

    window_size = len(training_config["loss_weights"])

    train_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["train"],
        sequence_names=sequence_names["train"],
        batch_size=training_config["batch_size"],
        window_size=window_size,
    )

    val_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["val"],
        sequence_names=sequence_names["val"],
        batch_size=training_config["batch_size"],
        window_size=window_size,
    )
    test_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["test"],
        sequence_names=sequence_names["test"],
        batch_size=training_config["batch_size"],
        window_size=window_size,
    )

    model = Model(
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        test_batcher=test_batcher,
        learning_rate=training_config["learning_rate"],
        epochs=training_config["epochs"],
        validation_steps=training_config["validation_steps"],
        optimizer=training_config["optimizer"]
    )

    model.train()
    # model.eval()
    model.save(dir_path=config["model_dir"])


if __name__ == "__main__":
    main()
