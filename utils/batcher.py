from utils.data_handlers.data_handler import DataHandler


class Batcher:
    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int = 64,
    ):

        self.data_handler = data_handler
        self.batch_size = batch_size

        # TODO refactor (好像直接整個移過來, 不需要在分 3 個多餘)
        self.windows = {}
        self.windows["train"] = self.data_handler.get_windows_by_split("train")
        self.windows["val"] = self.data_handler.get_windows_by_split("val")
        self.windows["test"] = self.data_handler.get_windows_by_split("test")

        if True:
            for split in self.windows:
                self.shuffle_data(split)

        self.current_index = 0
        self.current_split = "train"

    def get_num_window(self, split):
        return len(self.windows[split])

    def set_split(self, split="train"):
        if split not in self.windows:
            raise ValueError(f"Unknown split: {split}")
        self.current_split = split
        self.reset(self.current_split)

    def get_next(self):

        splitted_windows = self.windows[self.current_split]

        if self.current_index >= len(splitted_windows):
            self.reset(self.current_split)

        batch_windows = splitted_windows[
            self.current_index : self.current_index + self.batch_size
        ]

        data = self.data_handler.get_data_by_batch_windows(batch_windows)

        self.current_index += self.batch_size

        return data  # subject_ids, template_pcds, audios, label_pcds

    def reset(self, split: str):
        self.current_index = 0
        if True:
            self.shuffle_data(split)

    def shuffle_data(self, split: str):
        import random

        random.shuffle(self.windows[split])
