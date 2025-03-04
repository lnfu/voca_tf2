# TODO 使用 keras progress bar
# TODO 加上從中間開始指定 mesh 輸出影片

import numpy as np
import tensorflow as tf

import sys
import logging
import meshio

from scipy.io import wavfile

from utils.config import load_config
from utils.data_handlers.audio_handler import AudioHandler
from utils.batcher import Batcher
from utils.mesh.mesh_processor import MeshProcessor
from utils.inference import Inference

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

    template: meshio.Mesh = meshio.read(filename="data/FLAME_sample.ply", file_format="ply")

    audio_handler = AudioHandler(raw_path="data/audio/sample.wav")
    processed_audio = audio_handler.get_processed_data()["subject"]["sequence"]

    inference = Inference(config["model_dir"])
    delta_pcds = inference.predict_delta_pcds(0, processed_audio)

    num_frames = processed_audio.shape[0]
    assert num_frames == delta_pcds.shape[0]  # TODO
    mesh_processor = MeshProcessor(delta_pcds=delta_pcds, template=template)
    mesh_processor.save_to_obj_files(dir_path=config["output_dir"])
    mesh_processor.render_to_video(dir_path=config["output_dir"])


if __name__ == "__main__":
    main()
