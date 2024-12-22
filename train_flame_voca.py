import tensorflow as tf

import sys
import logging

from utils.flame_voca_model import FlameVocaModel
from utils.batcher import Batcher
from utils.data_handlers.data_handler import DataHandler
from utils.config import load_config

print(tf.config.list_physical_devices("GPU").__len__() > 0)  # 是否有使用 GPU

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    data_handler = DataHandler(
        audio_raw_path=config["data"]["audio"]["train"]["raw"],
        audio_processed_path=config["data"]["audio"]["train"]["processed"],
        mesh_dir_path=config["data"]["mesh"],
    )

    subject_names = config["subjects"]
    sequence_names = config["sequences"]

    # TODO k-fold
    train_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["train"],
        sequence_names=sequence_names["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
    )
    val_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["val"],
        sequence_names=sequence_names["val"],
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    test_batcher = Batcher(
        data_handler=data_handler,
        subject_names=subject_names["test"],
        sequence_names=sequence_names["test"],
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    model = FlameVocaModel(
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        test_batcher=test_batcher,
        # 包含 pos, vel, acc 三種 loss
        loss_weights=config["training"]["loss_weights"],
        optimizer=config["training"]["optimizer"],  # Adam
        learning_rate=config["training"]["learning_rate"],
        epochs=config["training"]["epochs"],
        # 0 表示不做 validation
        validation_freq=config["training"]["validation_freq"],
        reset=config["training"]["reset"],  # 是否從頭訓練 (否則會讀取 checkpoint)
        checkpoint_dir_path=config["checkpoint_dir"]
    )

    try:
        model.train()
    except KeyboardInterrupt:
        logging.warning("強制結束")
    finally:
        model.eval()
        print(f"Training tag: {model.save(dir_path=config["model_dir"])}")


if __name__ == "__main__":
    main()
