import numpy as np

import pickle
import logging


class IndexHandler:

    def __init__(
        self,
        subject_names: dict[str, list[str]],
        sequence_names: dict[str, list[str]],
    ):
        self.subject_names = subject_names
        self.sequence_names = sequence_names
        self.window_size = 2  # 考慮 velocity loss 需要到 2, 也許之後想要算 acceleration loss 就要用到　3

        self.generate_windows()

    def generate_windows(self):
        # 每個 window 都是一個 dict()
        # window["subject"]
        # window["sequence"]
        # window["indices"] = [?, ?]

        splits = ["train", "val", "test"]

        # TODO refactor .npy 檔案內容 (拋棄 local_index, 但是要先檢查它沒有用處)
        indices_by_subject_and_sequence = pickle.load(
            open("data/subj_seq_to_idx.pkl", "rb")
        )

        self.windows = {}

        for split in splits:
            subject_names = self.subject_names[split]
            sequence_names = self.sequence_names[split]

            self.windows[split] = []

            for subject_name in subject_names:
                for sequence_name in sequence_names:
                    if subject_name not in indices_by_subject_and_sequence:
                        logging.debug(
                            f"Index 資料不存在 Subject = {subject_name}, Sequence = *"
                        )
                        continue

                    if (
                        sequence_name
                        not in indices_by_subject_and_sequence[subject_name]
                    ):
                        logging.debug(
                            f"Index 資料不存在 Subject = {subject_name}, Sequence = {sequence_name}"
                        )
                        continue

                    raw_indices = indices_by_subject_and_sequence[subject_name][
                        sequence_name
                    ]
                    num_raw_indices = len(raw_indices)

                    indices = []
                    for frame_index, mesh_index in raw_indices.items():
                        indices.append((frame_index, mesh_index))

                    for i in range(len(indices)):
                        window_indices = [None] * self.window_size
                        for j in range(self.window_size):
                            k = i + j
                            if k >= len(indices):
                                continue
                            window_indices[j] = indices[k]

                        window = {}
                        window["subject"] = subject_name
                        window["sequence"] = sequence_name
                        window["indices"] = window_indices
                        self.windows[split].append(window)

