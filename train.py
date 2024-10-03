import tensorflow as tf

import sys
import logging

from utils.config import load_config

from utils.data_handlers.data_handler import DataHandler
from utils.batcher import Batcher

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    subject_names = {
        "train": [
            "FaceTalk_170728_03272_TA",
            "FaceTalk_170904_00128_TA",
            "FaceTalk_170725_00137_TA",
            "FaceTalk_170915_00223_TA",
            "FaceTalk_170811_03274_TA",
            "FaceTalk_170913_03279_TA",
            "FaceTalk_170904_03276_TA",
            "FaceTalk_170912_03278_TA",
        ],
        "val": ["FaceTalk_170811_03275_TA", "FaceTalk_170908_03277_TA"],
        "test": [
            "FaceTalk_170809_00138_TA",
            "FaceTalk_170731_00024_TA",
        ],
    }

    sequence_names = {
        "train": [
            "sentence01",
            "sentence02",
            "sentence03",
            "sentence04",
            "sentence05",
            "sentence06",
            "sentence07",
            "sentence08",
            "sentence09",
            "sentence10",
            "sentence11",
            "sentence12",
            "sentence13",
            "sentence14",
            "sentence15",
            "sentence16",
            "sentence17",
            "sentence18",
            "sentence19",
            "sentence20",
            "sentence21",
            "sentence22",
            "sentence23",
            "sentence24",
            "sentence25",
            "sentence26",
            "sentence27",
            "sentence28",
            "sentence29",
            "sentence30",
            "sentence31",
            "sentence32",
            "sentence33",
            "sentence34",
            "sentence35",
            "sentence36",
            "sentence37",
            "sentence38",
            "sentence39",
            "sentence40",
        ],
        "val": [
            "sentence21",
            "sentence22",
            "sentence23",
            "sentence24",
            "sentence25",
            "sentence26",
            "sentence27",
            "sentence28",
            "sentence29",
            "sentence30",
            "sentence31",
            "sentence32",
            "sentence33",
            "sentence34",
            "sentence35",
            "sentence36",
            "sentence37",
            "sentence38",
            "sentence39",
            "sentence40",
        ],
        "test": [
            "sentence21",
            "sentence22",
            "sentence23",
            "sentence24",
            "sentence25",
            "sentence26",
            "sentence27",
            "sentence28",
            "sentence29",
            "sentence30",
            "sentence31",
            "sentence32",
            "sentence33",
            "sentence34",
            "sentence35",
            "sentence36",
            "sentence37",
            "sentence38",
            "sentence39",
            "sentence40",
        ],
    }

    data_handler = DataHandler(
        subject_names=subject_names, sequence_names=sequence_names
    )

    batcher = Batcher(data_handler=data_handler)

    batcher.set_split("train")
    meshes, audios, subject_ids, templates = batcher.get_next()
    print(len(meshes))
    print(len(audios))
    print(len(subject_ids))
    print(len(templates))


if __name__ == "__main__":
    main()
