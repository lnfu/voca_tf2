import numpy as np
import tensorflow as tf

import os
import cv2
import logging
import meshio

from .mesh_render import MeshRenderer
from ..common import log_execution


class MeshProcessor:
    def __init__(
        self, pcds=None, delta_pcds=None, template: meshio.Mesh = None
    ) -> None:

        if pcds is not None:
            triangles = []
            with open(
                os.path.join(os.path.dirname(__file__), "triangles.txt"), "r"
            ) as file:
                for line in file:
                    triangles.append(
                        list(
                            map(lambda x: int(x) - 1, line.split(" ")[1:])
                        )  # 1-index (.obj format) -> 0-index (meshio)
                    )
            faces = [("triangle", triangles)]
        elif delta_pcds is not None and template is not None:
            faces = template.cells
            pcds = delta_pcds + template.points
        else:
            raise ValueError(
                "You must provide either 'pcds' or both 'delta_pcds' and 'template'."
            )

        self.meshes = [meshio.Mesh(points=pcd, cells=faces) for pcd in pcds]
        centers = np.mean(pcds, axis=1)  # (?, 3)
        self.center = np.mean(centers, axis=0)  # (3, )

    @log_execution
    def render_to_video(self, dir_path: str):
        mesh_renderer = MeshRenderer()
        progbar = tf.keras.utils.Progbar(self.num_frames)
        # save
        # with open("output/sample.mp4", "w") as f:
        video_writer = cv2.VideoWriter(
            os.path.join(dir_path, "sample.mp4"),  # TODO
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

    @log_execution
    def save_to_obj_files(self, dir_path: str):
        progbar = tf.keras.utils.Progbar(self.num_frames)
        for i, mesh in enumerate(self.meshes):
            mesh.write(
                os.path.join(dir_path, "meshes", "%05d.obj" % i), file_format="obj"
            )
            progbar.update(i + 1)

    @property
    def num_frames(self):
        return len(self.meshes)
