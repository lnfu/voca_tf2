# TODO 把 psbody.mesh 用這個完整取代 (psbody.mesh 有太多功能我不需要)
# TODO 自訂儲存路徑

import numpy as np
import tensorflow as tf

import os
import cv2
import logging

from .mesh_render import MeshRenderer
from psbody.mesh import Mesh


class MeshProcessor:
    def __init__(self, pcds=None, delta_pcds=None, template: Mesh=None) -> None:

        if pcds is not None:
            faces = []
            with open("data/faces.txt", "r") as face_file:
                for line in face_file:
                    faces.append(line.split(" "))
        elif delta_pcds is not None and template is not None:
            faces = template.f
            pcds = delta_pcds + template.v
        else:
            raise ValueError("You must provide either 'pcds' or both 'delta_pcds' and 'template'.")

        self.meshes = np.array([Mesh(pcd, faces) for pcd in pcds])
        centers = np.mean(pcds, axis=1)  # (?, 3)
        self.center = np.mean(centers, axis=0)  # (3, )

    def render_to_video(self):

        mesh_renderer = MeshRenderer()
        progbar = tf.keras.utils.Progbar(self.num_frames)
        # save
        # with open("output/sample.mp4", "w") as f:
        video_writer = cv2.VideoWriter(
            "outputss/sample.mp4",  # TODO
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (800, 800),
            True,
        )

        for i, mesh in enumerate(self.meshes):
            image = mesh_renderer.render_mesh_to_image(mesh=mesh, center=self.center)
            video_writer.write(image=image)
            progbar.update(i + 1)
        video_writer.release()
        logging.info("影片處理完成!")

    def save_to_obj_files(self):
        progbar = tf.keras.utils.Progbar(self.num_frames)

        for i, mesh in enumerate(self.meshes):
            mesh.write_obj(os.path.join("outputss/meshes/", "%05d.obj" % i))  # TODO
            progbar.update(i + 1)
        logging.info("OBJ files 存檔完成!")

    @property
    def num_frames(self):
        return self.meshes.shape[0]
