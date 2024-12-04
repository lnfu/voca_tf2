import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

print(tf.config.list_physical_devices("GPU").__len__() > 0)  # 是否有使用 GPU

import sys
import logging

from utils.config import load_config, get_data_config, get_training_config

from utils.data_handlers.data_handler import DataHandler
from utils.batcher import Batcher
from utils.flame_voca_model import FlameVocaModel

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
        mesh_path=data_config["path"]["mesh"],
    )

    subject_names = data_config["subjects"]
    sequence_names = data_config["sequences"]

    window_size = len(training_config["loss_weights"])

    train_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["train"],
        sequence_names=sequence_names["train"],
        shuffle=True,
    )

    val_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["val"],
        sequence_names=sequence_names["val"],
    )

    test_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["test"],
        sequence_names=sequence_names["test"],
    )

    model = FlameVocaModel(
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        test_batcher=test_batcher,
        learning_rate=training_config["learning_rate"],
        epochs=training_config["epochs"],
        validation_freq=training_config["validation_freq"],
        optimizer=training_config["optimizer"],
    )

    try:
        model.train()
    except KeyboardInterrupt:
        logging.warning("強制結束")
    finally:
        model.eval()
        model.save(dir_path=config["model_dir"])


if __name__ == "__main__":
    main()
