import numpy as np

import pickle


class PointCloudHandler:
    def __init__(self):
        self.load_template_pcds()
        self.load_pcds()

    def load_template_pcds(self):
        self.template_pcds = pickle.load(
            open("data/templates.pkl", "rb"), encoding="latin1"
        )

    def load_pcds(self):
        self.pcds = np.load("data/data_verts.npy", mmap_mode="r")
