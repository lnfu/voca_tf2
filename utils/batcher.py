import numpy as np
import tensorflow as tf

from utils.data_handlers.data_handler import DataHandler


class Batcher:

    def __init__(
        self,
        data_handler: DataHandler,
        subject_names: list[str],
        sequence_names: list[str],
        shuffle: bool = False,
        batch_size: int = 64,
    ):

        self.shuffle = shuffle

        self.data = []

        if subject_names == None:
            return

        if sequence_names == None:
            return

        for subject_name in subject_names:
            for sequence_name in sequence_names:
                if data_handler.is_subject_sequence_pair_valid(
                    subject_name, sequence_name
                ):
                    num_frame = data_handler.get_num_frame(
                        subject_name, sequence_name)

                    data_audio = data_handler.audio_processed_data[subject_name][sequence_name]
                    data_pcd = data_handler.pcd_data[subject_name][sequence_name]

                    data_ = [
                        {
                            "subject_id": subject_names.index(subject_name),
                            "audio": data_audio[i: i + batch_size],
                            "pcd": data_pcd[i: i + batch_size]
                        }
                        for i in range(0, num_frame, batch_size)
                    ]

                    self.data.extend(data_)

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
        self.current_batch_index += 1

        assert len(data_["audio"]) == len(data_["pcd"])
        return (
            np.full(len(data_["audio"]), data_["subject_id"]),
            data_["audio"],
            data_["pcd"]
        )

    def shuffle_data(self):
        import random

        random.shuffle(self.data)
