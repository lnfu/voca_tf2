import numpy as np

import logging

from .index_handler import IndexHandler
from .audio_handler import AudioHandler
from .pcd_handler import PointCloudHandler


class DataHandler:
    def __init__(self):

        self.audio_data_handler = AudioHandler()
        self.pcd_data_handler = PointCloudHandler()
        self.index_data_handler = IndexHandler()

        self.audio_processed_data = self.audio_data_handler.get_processed_training_data()

        # TODO 拆成 method
        self.data = {}
        for (
            subject_name,
            subject_data,
        ) in self.index_data_handler.get_index_windws().items():
            if subject_name not in self.audio_processed_data:
                logging.warning(f"音訊資料不存在 Subject = {subject_name}, Sequence = *, Frame = *")
                continue

            self.data[subject_name] = {}
            for sequence_name, sequence_data in subject_data.items():

                if sequence_name not in self.audio_processed_data[subject_name]:
                    logging.warning(f"音訊資料不存在 Subject = {subject_name}, Sequence = {sequence_name}, Frame = *")
                    continue

                self.data[subject_name][sequence_name] = []

                for index_window in sequence_data:
                    window = [None] * len(index_window)
                    for i, index in enumerate(index_window):
                        if index is None:
                            continue
                        frame_index, pcd_index = index
                        if frame_index >= self.audio_processed_data[subject_name][sequence_name].shape[0]:
                            logging.error(
                                f"音訊資料不存在 subject = {subject_name}, sequence = {sequence_name}, frame = {frame_index}"
                            )
                            continue
                        if pcd_index >= self.pcd_data_handler.get_num_pcds():
                            logging.warning(f"Point Cloud 不存在, index = {pcd_index}")
                            continue

                        window_item = {}
                        window_item["subject"] = subject_name
                        window_item["sequence"] = sequence_name
                        window_item["audio"] = self.audio_processed_data[subject_name][sequence_name][frame_index]
                        window_item["pcd_index"] = pcd_index
                        window[i] = window_item
                    self.data[subject_name][sequence_name].append(window)

    # 會找出 pcd_index 對應的 pcd, 並且把 4 個 key 分成 4 個 list
    def unpack_data(self, batch_data_index_only: list[list]) -> tuple:

        batch_audio = [[window_item["audio"] for window_item in window] for window in batch_data_index_only]
        batch_pcd = [
            [self.pcd_data_handler.get_pcd_by_index(window_item["pcd_index"]) for window_item in window]
            for window in batch_data_index_only
        ]

        batch_subject_name = []
        batch_template_pcd = []

        for window in batch_data_index_only:
            subjects = [window_item["subject"] for window_item in window]
            if len(set(subjects)) > 1:
                logging.error(f"Subjects in window are not the same: {subjects}")

            batch_subject_name.append(subjects[0])
            batch_template_pcd.append(self.pcd_data_handler.get_template_pcd_by_subject_name(subjects[0]))

        return (
            batch_subject_name,
            np.array(batch_audio),
            np.array(batch_template_pcd),
            np.array(batch_pcd),
        )

    # TODO: deprecated?
    def get_pcds_by_subject_and_sequence(self, subject_name: str, sequence_name: str):
        indices = self.index_data_handler.get_indices_by_subject_and_sequence(
            subject_name=subject_name, sequence_name=sequence_name
        )
        return [self.pcd_data_handler.get_pcd_by_index(pcd_index) for frame_index, pcd_index in indices.items()]
