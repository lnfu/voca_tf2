import tensorflow as tf

import sys
import pickle
import logging

from utils.config import load_config
from utils.data_handlers.audio_data_handler import AudioDataHandler
# from utils.data_handlers.mesh_data_handler import MeshDataHandler
# from utils.batcher import Batcher

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    audio_data_handler = AudioDataHandler()

    test = audio_data_handler.get_processed_data()

    for subject_name, subject_data in test.items():
        for sequence_name, sequence_data in subject_data.items():
            print(subject_name, sequence_name, sequence_data.shape)

    # mesh_data_handler = MeshDataHandler()

    # batcher = Batcher(
    #     mesh_data_handler=mesh_data_handler,
    #     splitted_indices={"train": [], "val": [], "test": []},
    # )


if __name__ == "__main__":
    main()
