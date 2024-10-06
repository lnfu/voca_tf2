import numpy as np

import pickle


class PointCloudHandler:
    def __init__(self):
        self.load_template_pcds()
        self.load_pcds()

    def load_template_pcds(self):
        self.template_pcds = pickle.load(open("data/templates.pkl", "rb"), encoding="latin1")

    def load_pcds(self):
        self.pcds = np.load("data/data_verts.npy", mmap_mode="r")  # (123341, 5023, 3)

    def get_template_pcd_by_subject_name(self, subject_name: str):
        return self.template_pcds[subject_name]

    def get_pcd_by_index(self, index: int):
        return self.pcds[index]

    def get_num_pcds(self):
        return self.pcds.shape[0]
