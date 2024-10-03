from .index_handler import IndexHandler
from .audio_handler import AudioHandler
from .mesh_handler import MeshHandler

import logging


class DataHandler:
    def __init__(
        self,
        subject_names: dict[str, list[str]],
        sequence_names: dict[str, list[str]],
    ):
        self.subject_names = subject_names
        self.sequence_names = sequence_names

        self.audio_data_handler = AudioHandler()
        self.mesh_data_handler = MeshHandler()
        self.index_data_handler = IndexHandler(
            subject_names=subject_names, sequence_names=sequence_names
        )

        self.audio_processed_data = self.audio_data_handler.get_processed_data()

    def get_training_subject_id_by_name(self, name):
        try:
            return self.subject_names["train"].index(name)
        except ValueError:
            logging.error(f"Subject 不存在: {name}")
            return None

    def get_data_by_batch_windows(self, batch_windows: list[list]) -> tuple:

        filtered_batch_windows = [
            window for window in batch_windows if None not in window["indices"]
        ]

        mesh_indices = []

        meshes = []
        audios = []
        subject_ids = []
        templates = []

        for window in filtered_batch_windows:
            subject_name = window["subject"]
            sequence_name = window["sequence"]
            if subject_name not in self.audio_processed_data:
                logging.warning(
                    f"音訊資料不存在 Subject = {subject_name}, Sequence = *"
                )
                continue
            if sequence_name not in self.audio_processed_data[subject_name]:
                logging.warning(
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
                logging.warning(
                    f"音訊資料不正確 (frame_index 超出範圍) Subject = {subject_name}, Sequence = {sequence_name}"
                )
                continue

            for frame_index, mesh_index in window["indices"]:
                mesh_indices.append(mesh_index)
                audios.append(
                    self.audio_processed_data[subject_name][sequence_name][frame_index]
                )
                subject_ids.append(self.get_training_subject_id_by_name(subject_name))
                templates.append(
                    self.mesh_data_handler.templates[subject_name]
                )  # TODO check 存在

        meshes = self.mesh_data_handler.meshes[mesh_indices]

        return meshes, audios, subject_ids, templates

    def get_windows_by_split(self, split):
        return self.index_data_handler.windows[split]
