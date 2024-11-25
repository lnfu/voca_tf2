# TODO 使用 keras progress bar
# TODO 加上從中間開始指定 mesh 輸出影片

import numpy as np
import tensorflow as tf

import os
import sys
import time
import meshio
import logging

from scipy.io import wavfile

from utils.config import load_config
from utils.data_handlers.audio_handler import AudioHandler
from utils.batcher import Batcher
from utils.mesh.mesh_processor import MeshProcessor
from utils.inference import Inference
from utils.flame import Flame

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

    audio_handler = AudioHandler(raw_path="data/audio/shakira.wav")
    processed_audio = audio_handler.get_processed_data()["subject"]["sequence"]

    logging.info("正在載入 VOCA 模型...")
    model = tf.keras.models.load_model(config["model_dir"])
    logging.info("VOCA 模型成功載入!")
    np.save("shape.npy", model.trainable_variables[-1])

    # prediction
    flame_params = model.predict(
        [
            processed_audio,
        ]
    )

    print(type(flame_params))
    np.save("temp.npy", flame_params)
    exit(0)

    start_time = time.time()  # TODO
    pred_pcds = tf.map_fn(lambda x: Flame.calculate_pcd_by_param(x), flame_params)
    end_time = time.time()  # TODO
    # logging.info(f"Execution time: {end_time - start_time:.4f} seconds")  # TODO
    # exit(0)

    num_frames = processed_audio.shape[0]
    assert num_frames == flame_params.shape[0]  # TODO

    mesh_processor = MeshProcessor(pcds=pred_pcds)
    mesh_processor.save_to_obj_files(dir_path=config["output_dirs"]["mesh"])
    mesh_processor.render_to_video(dir_path=config["output_dirs"]["video"])


if __name__ == "__main__":
    main()
