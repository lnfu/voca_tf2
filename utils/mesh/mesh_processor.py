import tensorflow as tf

import os
import cv2
import meshio
import subprocess

from .mesh_render import MeshRenderer
from ..common import log_execution, check_and_create_directory


def get_flame_mesh_triangles():
    triangles = []
    with open(
        os.path.join(os.path.dirname(__file__), "triangles.txt"), "r"
    ) as file:
        for line in file:
            triangles.append(
                list(
                    map(lambda x: int(x) - 1, line.split(" "))
                )  # 1-index (.obj format) -> 0-index (meshio)
            )
    return triangles


class MeshProcessor:
    def __init__(
        self,
        pcds=None,
        delta_pcds=None,
        template_pcds=None,
        template: meshio.Mesh = None,
    ) -> None:

        faces = [("triangle", get_flame_mesh_triangles())]

        if pcds is not None:
            pass
        elif delta_pcds is not None and template_pcds is not None:
            pcds = delta_pcds + template_pcds
        elif delta_pcds is not None and template is not None:
            # TODO assert faces = template.cells
            pcds = delta_pcds + template.points
        else:
            raise ValueError(
                "You must provide either 'pcds' or both 'delta_pcds' and 'template'."
            )

        self.meshes = [meshio.Mesh(points=pcd, cells=faces) for pcd in pcds]

    @log_execution
    def save_to_video(self, video_path: str):
        mesh_renderer = MeshRenderer()
        progbar = tf.keras.utils.Progbar(self.num_frames)

        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (800, 800),
            True,
        )

        for i, mesh in enumerate(self.meshes):
            image = mesh_renderer.render_mesh_to_image(mesh=mesh)
            video_writer.write(image=image)
            progbar.update(i + 1)
        video_writer.release()

    def merge_video_audio(self, video_path: str, audio_path: str, output_path: str):
        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ])

    @log_execution
    def save_to_obj_files(self, dir_path: str):
        check_and_create_directory(dir_path)
        progbar = tf.keras.utils.Progbar(self.num_frames)
        for i, mesh in enumerate(self.meshes):
            mesh.write(os.path.join(dir_path, "%05d.obj" %
                       i), file_format="obj")
            progbar.update(i + 1)

    @property
    def num_frames(self):
        return len(self.meshes)
