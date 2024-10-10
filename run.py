# TODO 加上音訊
# TODO 使用 keras progress bar
# TODO 加上從中間開始指定 mesh 輸出影片

import numpy as np
import tensorflow as tf

import os
import sys
import cv2
import logging

from psbody.mesh import Mesh
from scipy.io import wavfile

from utils.config import load_config
from utils.data_handlers.audio_handler import AudioHandler
from utils.batcher import Batcher
from utils.model import custom_loss
from utils.mesh.mesh_processor import MeshProcessor

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    template = Mesh(filename="data/FLAME_sample.ply")
    print(template.v.shape)

    sample_rate, audio = wavfile.read("data/sample.wav")

    audio_handler = AudioHandler()

    processed_audio = audio_handler.batch_process(
        raw_data={
            "sample": {
                "sample": {
                    "sample_rate": sample_rate,
                    "audio": audio,
                }
            }
        }
    )["sample"]["sample"]

    num_frames = processed_audio.shape[0]

    logging.info("正在載入 VOCA 模型...")
    tf.keras.utils.get_custom_objects()["custom_loss"] = custom_loss
    model = tf.keras.models.load_model(config["model_dir"])
    logging.info("VOCA 模型成功載入!")

    delta_pcds = model.predict(
        [
            np.repeat(0, num_frames, axis=0),
            processed_audio,
        ]
    )

    logging.info("預測完成, 開始寫入資料...")

    assert num_frames == delta_pcds.shape[0]  # TODO

    mesh_processor = MeshProcessor(delta_pcds=delta_pcds, template=template)

    mesh_processor.save_to_obj_files()
    mesh_processor.render_to_video()


if __name__ == "__main__":
    main()
