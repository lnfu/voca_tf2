import tensorflow as tf

import os
import logging
import sys
import numpy as np

from utils.common import check_and_create_directory
from utils.flame import Flame
from utils.mesh.mesh_processor import MeshProcessor
from utils.data_handlers.audio_handler import AudioHandler
from utils.config import load_config

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    audio_handler = AudioHandler(
        raw_path=config["data"]["audio"]["run"]["raw"])
    processed_audio = audio_handler.get_processed_data()["subject"]["sequence"]

    # 載入模型
    train_tag = input("Enter training tag: ")
    model = tf.keras.models.load_model(
        os.path.join(config["model_dir"], train_tag))

    # 根據音訊預測臉部 FLAME 參數 (415)
    flame_params = model.predict([processed_audio])

    # 儲存 FLAME 參數
    check_and_create_directory("outputs")  # TODO
    np.save(config["output"]["flame_param"], flame_params)
    num_frames = processed_audio.shape[0]
    assert num_frames == flame_params.shape[0]

    # 獲得臉部點雲 (5023)
    pred_pcds = tf.map_fn(
        lambda x: Flame.calculate_pcd_by_param(x), flame_params)

    mesh_processor = MeshProcessor(pcds=pred_pcds)
    mesh_processor.save_to_obj_files(dir_path=config["output"]["mesh_dir"])
    mesh_processor.save_to_video(video_path=config["output"]["video_only"])
    mesh_processor.merge_video_audio(
        video_path=config["output"]["video_only"],  # 視訊
        audio_path=config["data"]["audio"]["run"]["raw"],  # 音訊
        output_path=config["output"]["video"]  # 視訊 + 音訊
    )


if __name__ == "__main__":
    main()
