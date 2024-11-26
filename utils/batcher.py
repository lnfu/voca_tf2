import numpy as np
import tensorflow as tf

from utils.data_handlers.data_handler import DataHandler


class Batcher:

    def __init__(
        self,
        data_handler: DataHandler,
        subject_names: set[str],  # 目前只會挑一個 subject
        sequence_names: set[str],
        shuffle: bool = False,  # 只會打亂句子
    ):

        self.shuffle = shuffle

        self.data = []

        for subject_name in subject_names:
            for sequence_name in sequence_names:
                if data_handler.is_subject_sequence_pair_valid(
                    subject_name, sequence_name
                ):
                    data_ = {}
                    data_["num_frame"] = data_handler.get_num_frame(
                        subject_name, sequence_name
                    )
                    data_["audio"] = data_handler.audio_processed_data[subject_name][
                        sequence_name
                    ]
                    data_["pcd"] = data_handler.pcd_data[subject_name][sequence_name]
                    self.data.append(data_)

        self.reset()

    def reset(self):
        self.current_batch_index = 0
        if self.shuffle:
            self.shuffle_data()

    def get_num_batches(self):
        return len(self.data)

    def get_next(self):

        if self.current_batch_index >= self.get_num_batches():
            self.reset()
        
        data_ = self.data[self.current_batch_index]

        return data_["num_frame"], data_["audio"], data_["pcd"]

    def shuffle_data(self):
        import random

        random.shuffle(self.data)
