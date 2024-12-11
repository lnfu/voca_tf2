import tensorflow as tf
import os
import cv2
import meshio

from utils.mesh.mesh_render import MeshRenderer
from utils.config import load_config, get_data_config
from utils.common import check_and_create_directory, subject_names, sequence_names


from PIL import Image

def main():
    mesh_renderer = MeshRenderer()

    config = load_config("config.yaml")
    data_config = get_data_config(config)

    mesh_dir_path=data_config["path"]["mesh"]
    check_and_create_directory(str(os.path.join(config["output_dirs"]["video"], "gt")))

    # subject_names = data_config["subjects"]["train"] # TODO
    # sequence_names = data_config["sequences"]["train"] # TODO

    mesh_renderer = MeshRenderer()

    for subject_name in subject_names:
        check_and_create_directory(str(os.path.join(config["output_dirs"]["video"], "gt", subject_name)))    

        for sequence_name in sequence_names:

            dir_path = os.path.join(mesh_dir_path, subject_name, sequence_name) # TODO rename var
            if not os.path.exists(dir_path):
                continue
            
            meshes = {}
            for ply_file_path in os.listdir(dir_path):
                if ply_file_path.endswith(".ply"):
                    meshes[ply_file_path] = meshio.read(
                        os.path.join(dir_path, ply_file_path)
                    )
            num_frames = len(meshes)
            meshes = [mesh for _, mesh in sorted(meshes.items())]

            progbar = tf.keras.utils.Progbar(num_frames)

            video_writer = cv2.VideoWriter(
                os.path.join(config["output_dirs"]["video"], "gt", subject_name, f'{sequence_name}.mp4'),  # TODO
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (800, 800),
                True,
            )

            for i, mesh in enumerate(meshes):
                # image = mesh_renderer.render_mesh_to_image(mesh=mesh, center=self.center)
                image = mesh_renderer.render_mesh_to_image(mesh=mesh)
                video_writer.write(image=image)
                progbar.update(i + 1)
            video_writer.release()


if __name__ == "__main__":
    main()
