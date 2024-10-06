import numpy as np

import pickle
import logging


class IndexHandler:

    def __init__(self):
        self.max_window_size = 5  # TODO
        self.indices_by_subject_and_sequence = pickle.load(open("data/subj_seq_to_idx.pkl", "rb"))
        self.generate_indix_windows()

    # TODO: deprecated??
    def get_indices_by_subject_and_sequence(self, subject_name: str, sequence_name: str):
        # indices = [(frame_index, mesh_index), ...]

        if subject_name not in self.indices_by_subject_and_sequence:
            logging.warning(f"Index 資料不存在 Subject = {subject_name}, Sequence = *")
            return []
        if sequence_name not in self.indices_by_subject_and_sequence[subject_name]:
            logging.debug(f"Index 資料不存在 Subject = {subject_name}, Sequence = {sequence_name}")
            return []

        return self.indices_by_subject_and_sequence[subject_name][sequence_name]

    def get_index_windws(self):
        return self.index_windows

    def generate_indix_windows(self):

        def build_index_window(indices, start_idx, max_window_size):
            index_window = [None] * max_window_size
            for j in range(max_window_size):
                k = start_idx + j
                if k < len(indices):
                    index_window[j] = indices[k]
            return index_window

        self.index_windows = {}

        for subject_name, subject_data in self.indices_by_subject_and_sequence.items():
            self.index_windows[subject_name] = {}

            for sequence_name, sequence_data in subject_data.items():

                indices = list(sequence_data.items())
                self.index_windows[subject_name][sequence_name] = [
                    build_index_window(indices, i, self.max_window_size) for i in range(len(indices))
                ]
