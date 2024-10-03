import numpy as np

import pickle
import logging


class MeshHandler:
    def __init__(self):
        self.load_templates()
        self.load_meshes()

    def load_templates(self):
        self.templates = pickle.load(
            open("data/templates.pkl", "rb"), encoding="latin1"
        )

    def load_meshes(self):
        self.meshes = np.load("data/data_verts.npy", mmap_mode="r")
