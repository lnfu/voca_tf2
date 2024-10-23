import numpy as np

from utils.data_handlers.data_handler import DataHandler


class Batcher:
    def get_num_data(self):
        return len(self.data)

    def get_num_batches(self):
        return self.get_num_data() // self.batch_size

    def get_num_subjects(self):
        return len(self.subject_names)

    def get_subject_id_by_name(self, name: str, default_value: int = None):
        try:
            return self.subject_names.index(name)
        except ValueError:
            if default_value is not None:
                return default_value
            return np.random.randint(0, len(self.subject_names))

    def __init__(
        self,
        data_handler: DataHandler,
        subject_names: set[str],
        sequence_names: set[str],
        batch_size: int = 64,
        window_size: int = 1,  # 主要計算 loss (velocity) 才會用到
        shuffle: bool = True,
    ):

        self.data_handler = data_handler
        self.batch_size = batch_size
        self.window_size = window_size
        self.subject_names = subject_names

        self.shuffle = shuffle

        # filter
        filtered = {
            subject_name: {
                sequence_name: [
                    window[
                        : self.window_size
                    ]  # 每個 window 只取前 self.window_size 並且不能有 None
                    for window in sequence_data
                    if all(
                        window_item is not None
                        for window_item in window[: self.window_size]
                    )
                ]
                for sequence_name, sequence_data in subject_data.items()
                if sequence_name in sequence_names
            }
            for subject_name, subject_data in self.data_handler.data.items()
            if subject_name in subject_names
        }

        # flatten
        self.data = [
            window
            for subject_data in filtered.values()
            for sequence_data in subject_data.values()
            for window in sequence_data
        ]

        self.reset()

    def get_next(self):

        if self.current_index >= len(self.data):
            self.reset()

        batch_data_index_only = self.data[
            self.current_index : self.current_index + self.batch_size
        ]  # 這種寫法右界超過也沒關係

        batch_subject_name, batch_template_pcd, batch_pcd, batch_audio = (
            self.data_handler.unpack_data(batch_data_index_only)
        )

        batch_subject_id = [
            self.get_subject_id_by_name(subject_name)  # random if not valid
            # self.get_subject_id_by_name(subject_name, 0) # 0 if not valid
            for subject_name in batch_subject_name
        ]

        self.current_index += self.batch_size  # 更新 current_index

        return (
            np.array(batch_subject_id),  # (None, 64, 8)
            batch_template_pcd,  # (None, 5023, 3)
            batch_pcd,  # (None, 5023, 3, 2)
            batch_audio,  # (None, 16, 29, 2)
        )

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        import random

        random.shuffle(self.data)
