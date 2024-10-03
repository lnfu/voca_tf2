import numpy as np
import tensorflow as tf

import os
import sys
import logging

from psbody.mesh import Mesh
from scipy.io import wavfile

from utils.config import load_config
from utils.data_handlers.audio_handler import AudioHandler
from utils.batcher import Batcher
from utils.model import custom_loss


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
    model = tf.keras.models.load_model("models")
    logging.info("VOCA 模型成功載入!")

    pred_pcds = model.predict(
        [
            np.repeat(0, num_frames, axis=0),
            np.repeat(np.expand_dims(template.v, axis=0), num_frames, axis=0),
            processed_audio,
        ]
    )

    print(pred_pcds.shape)

    # render
    for i, pcd in enumerate(pred_pcds):
        mesh = Mesh(pcd, template.f)
        mesh.write_obj(os.path.join("outputs/meshes/", "%05d.obj" % i))

if __name__ == "__main__":
    main()
