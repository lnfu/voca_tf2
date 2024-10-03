import numpy as np

import logging

from .index_handler import IndexHandler
from .audio_handler import AudioHandler
from .pcd_handler import PointCloudHandler


class DataHandler:
    def __init__(
        self,
        subject_names: dict[str, list[str]],
        sequence_names: dict[str, list[str]],
    ):
        self.subject_names = subject_names
        self.sequence_names = sequence_names

        self.audio_data_handler = AudioHandler()
        self.pcd_data_handler = PointCloudHandler()
        self.index_data_handler = IndexHandler(
            subject_names=subject_names, sequence_names=sequence_names
        )

        self.audio_processed_data = self.audio_data_handler.get_processed_training_data()

    def get_training_subject_id_by_name(self, name):
        try:
            return self.subject_names["train"].index(name)
        except ValueError:
            return np.random.randint(0, len(self.subject_names["train"]))

    def get_data_by_batch_windows(self, batch_windows: list[list]) -> tuple:

        filtered_batch_windows = [
            window for window in batch_windows if None not in window["indices"]
        ]

        pcd_indices = []

        label_pcds = []
        audios = []
        subject_ids = []
        template_pcds = []

        for window in filtered_batch_windows:
            subject_name = window["subject"]
            sequence_name = window["sequence"]
            if subject_name not in self.audio_processed_data:
                logging.debug(
                    f"音訊資料不存在 Subject = {subject_name}, Sequence = *"
                )
                continue
            if sequence_name not in self.audio_processed_data[subject_name]:
                logging.debug(
                    f"音訊資料不存在 Subject = {subject_name}, Sequence = {sequence_name}"
                )
                continue

            is_audio_valid = True
            for frame_index, _ in window["indices"]:
                if frame_index >= len(
                    self.audio_processed_data[subject_name][sequence_name]
                ):
                    is_audio_valid = False
                    break

            if not is_audio_valid:
                logging.debug(
                    f"音訊資料不正確 (frame_index 超出範圍) Subject = {subject_name}, Sequence = {sequence_name}"
                )
                continue

            for frame_index, pcd_index in window["indices"]:
                pcd_indices.append(pcd_index)
                audios.append(
                    self.audio_processed_data[subject_name][sequence_name][frame_index]
                )
                subject_ids.append(self.get_training_subject_id_by_name(subject_name))
                template_pcds.append(
                    self.pcd_data_handler.template_pcds[subject_name]
                )  # TODO check 存在

        label_pcds = self.pcd_data_handler.pcds[pcd_indices]
        return np.array(subject_ids), np.array(template_pcds), np.array(audios), label_pcds

    def get_windows_by_split(self, split):
        return self.index_data_handler.windows[split]
