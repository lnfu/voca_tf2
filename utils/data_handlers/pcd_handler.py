import os
import meshio
import logging

import numpy as np
import tensorflow as tf

from .. import common

# TODO subjects and sentences move to another place

pcds_file_name = "pcds.npy"

class PointCloudHandler:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path

    def get_processed_data(self, reset: bool = False):
        pcds = {}

        progbar = tf.keras.utils.Progbar(target=len(common.subject_names) * len(common.sequence_names))
        step = 0
        for subject in common.subject_names:
            pcds[subject] = {}

            for sentence in common.sequence_names:
                dir_path = os.path.join(self.mesh_path, subject, sentence)

                if not os.path.exists(dir_path):
                    continue

                pcds_file_path = os.path.join(dir_path, pcds_file_name)
                if not reset and os.path.exists(pcds_file_path):
                    pcds_ = self.get_pcds_from_npy(pcds_file_path)
                    logging.info("已處理過, 直接讀取")
                else:
                    logging.info("重新提取 Point Cloud Data...")
                    pcds_ = self.get_pcds_from_plys(dir_path)
                    np.save(pcds_file_path, pcds_, allow_pickle=False)

                pcds[subject][sentence] = pcds_

                progbar.update(step + 1)
                step += 1

        return pcds

    def get_pcds_from_npy(self, npy_file_path: str):
        return np.load(npy_file_path)

    def get_pcds_from_plys(self, dir_path: str):
        meshes = {}
        for ply_file_path in os.listdir(dir_path):
            if ply_file_path.endswith(".ply"):
                meshes[ply_file_path] = meshio.read(
                    os.path.join(dir_path, ply_file_path)
                )

        return np.stack(
            [mesh.points for _, mesh in sorted(meshes.items())]
        )  # per subject per sentence

