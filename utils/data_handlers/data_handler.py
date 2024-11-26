import numpy as np

import logging

from collections import defaultdict
from .audio_handler import AudioHandler
from .pcd_handler import PointCloudHandler
from .. import common


class DataHandler:
    def __init__(
        self,
        audio_raw_path,
        audio_processed_path,
        mesh_path,
    ):
        audio_data_handler = AudioHandler(
            raw_path=audio_raw_path, processed_path=audio_processed_path
        )
        self.audio_processed_data = audio_data_handler.get_processed_data()

        pcd_data_handler = PointCloudHandler(mesh_path=mesh_path)
        self.pcd_data = pcd_data_handler.get_processed_data()

        self.validate_data()

    def is_subject_sequence_pair_valid(
        self, subject_name: str, sequence_name: str
    ) -> bool:
        return self.valid[subject_name][sequence_name]

    def get_num_frame(self, subject_name: str, sequence_name: str) -> int:
        return self.num_frame_data[subject_name][sequence_name]

    def validate_data(self):
        self.valid = defaultdict(lambda: False)
        self.num_frame_data = defaultdict(lambda: -1)

        for subject_name in common.subject_names:
            self.valid[subject_name] = defaultdict(lambda: False)
            self.num_frame_data[subject_name] = defaultdict(lambda: -1)

            if subject_name not in self.audio_processed_data.keys():
                logging.warning(f"缺少音訊 subject={subject_name}, sentence=*")
                continue

            if subject_name not in self.pcd_data.keys():
                logging.warning(f"缺少點雲 subject={subject_name}, sentence=*")
                continue

            for sequence_name in common.sequence_names:

                if sequence_name not in self.audio_processed_data[subject_name].keys():
                    logging.warning(
                        f"缺少音訊 subject={subject_name}, sentence={sequence_name}"
                    )
                    continue

                if sequence_name not in self.pcd_data[subject_name].keys():
                    logging.warning(
                        f"缺少點雲 subject={subject_name}, sentence={sequence_name}"
                    )
                    continue

                # 目前遇到蠻多音訊比點雲多了一兩個 frame, 先考慮把音訊的最後忽略
                audio_processed_data_num_frame = self.audio_processed_data[
                    subject_name
                ][sequence_name].shape[0]
                pcd_data_num_frame = self.pcd_data[subject_name][sequence_name].shape[0]

                if audio_processed_data_num_frame != pcd_data_num_frame:
                    logging.warning(
                        f"音訊 {audio_processed_data_num_frame} 和點雲 {pcd_data_num_frame} 資料 frame 數量不符合 subject={subject_name}, sentence={sequence_name}"
                    )

                num_frame = min(audio_processed_data_num_frame, pcd_data_num_frame)
                self.audio_processed_data[subject_name][sequence_name] = (
                    self.audio_processed_data[subject_name][sequence_name][:num_frame]
                )
                self.pcd_data[subject_name][sequence_name] = self.pcd_data[
                    subject_name
                ][sequence_name][:num_frame]
                self.valid[subject_name][sequence_name] = True
                self.num_frame_data[subject_name][sequence_name] = num_frame
