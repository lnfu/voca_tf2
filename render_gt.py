import tensorflow as tf

import os
import cv2
import meshio

from utils.mesh.mesh_render import MeshRenderer
from utils.config import load_config
from utils.common import check_and_create_directory, subject_names, sequence_names

RENDER_ALL = True


def main():
    global subject_names, sequence_names

    mesh_renderer = MeshRenderer()

    config = load_config("config.yaml")

    mesh_dir_path = config["data"]["mesh"]
    output_dir_path = config["output"]["gt_dir"]
    check_and_create_directory(output_dir_path)

    if not RENDER_ALL:
        subject_names = config["subjects"]["train"]
        sequence_names = config["sequences"]["train"]

    mesh_renderer = MeshRenderer()

    for sequence_name in sequence_names:
        for subject_name in subject_names:
            output_per_subject_dir_path = str(
                os.path.join(output_dir_path, subject_name))
            check_and_create_directory(output_per_subject_dir_path)

            # 來源 mesh 的位置
            mesh_per_subject_sequence_dir_path = os.path.join(
                mesh_dir_path, subject_name, sequence_name)
            if not os.path.exists(mesh_per_subject_sequence_dir_path):
                continue

            meshes = {}
            for ply_file_path in os.listdir(mesh_per_subject_sequence_dir_path):
                if ply_file_path.endswith(".ply"):
                    meshes[ply_file_path] = meshio.read(
                        os.path.join(
                            mesh_per_subject_sequence_dir_path, ply_file_path)
                    )
            num_frames = len(meshes)
            meshes = [mesh for _, mesh in sorted(meshes.items())]

            progbar = tf.keras.utils.Progbar(num_frames)

            video_writer = cv2.VideoWriter(
                os.path.join(output_per_subject_dir_path,
                             f'{sequence_name}.mp4'),
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (800, 800),
                True,
            )

            for i, mesh in enumerate(meshes):
                image = mesh_renderer.render_mesh_to_image(mesh=mesh)
                video_writer.write(image=image)
                progbar.update(i + 1)
            video_writer.release()


if __name__ == "__main__":
    main()
