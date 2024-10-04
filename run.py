import numpy as np
import tensorflow as tf

import os
import sys
import cv2
import logging
import trimesh
import pyrender

from psbody.mesh import Mesh
from scipy.io import wavfile

from utils.config import load_config
from utils.data_handlers.audio_handler import AudioHandler
from utils.batcher import Batcher
from utils.model import custom_loss

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    config = load_config("config.yaml")

    template = Mesh(filename="data/FLAME_sample.ply")
    print(template.v.shape)

    sample_rate, audio = wavfile.read("data/sample.wav")

    audio_handler = AudioHandler()

    processed_audio = audio_handler.batch_process(
        raw_data={
            "sample": {
                "sample": {
                    "sample_rate": sample_rate,
                    "audio": audio,
                }
            }
        }
    )["sample"]["sample"]

    num_frames = processed_audio.shape[0]

    logging.info("正在載入 VOCA 模型...")
    tf.keras.utils.get_custom_objects()["custom_loss"] = custom_loss
    model = tf.keras.models.load_model("models")
    logging.info("VOCA 模型成功載入!")

    pred_pcds = model.predict(
        [
            np.repeat(0, num_frames, axis=0),
            np.repeat(np.expand_dims(template.v, axis=0), num_frames, axis=0),
            processed_audio,
        ]
    )

    assert num_frames == pred_pcds.shape[0]

    # pred_pcds.shape = (?, 5023, 3)
    centers = np.mean(pred_pcds, axis=1)  # (?, 3)
    center = np.mean(centers, axis=0)  # (3, )

    # save
    # with open("output/sample.mp4", "w") as f:
    video_writer = cv2.VideoWriter("outputs/sample.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 800), True)
    for i, pcd in enumerate(pred_pcds):
        print(i)
        mesh = Mesh(pcd, template.f)
        mesh.write_obj(os.path.join("outputs/meshes/", "%05d.obj" % i))

        image = render_mesh_to_image(mesh=mesh, center=center)
        video_writer.write(image)
    video_writer.release()


def render_mesh_to_image(mesh, center, rotation=np.zeros(3)):

    camera_params = {
        "optical_center": [400.0, 400.0],
        "focal_length": [4754.97941935 / 2, 4754.97941935 / 2],  # TODO 數值???
    }

    frustum_params = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}

    # 建立場景
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[255, 255, 255])

    # 加入 camera
    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["focal_length"][0],
        fy=camera_params["focal_length"][1],
        cx=camera_params["optical_center"][0],
        cy=camera_params["optical_center"][1],
        znear=frustum_params["near"],
        zfar=frustum_params["far"],
    )
    scene.add(camera, pose=np.eye(4))

    # 加入光源
    light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.5)
    scene.add(light, pose=np.eye(4))

    mesh_copy = Mesh(mesh.v, mesh.f)

    # 圍繞 center 做旋轉 rotation
    mesh_copy.v[:] = cv2.Rodrigues(rotation)[0].dot((mesh_copy.v - center).T).T + center

    # trimesh 上色 (沒有色彩)
    tri_mesh = trimesh.Trimesh(
        vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=None
    )

    # pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene.add(render_mesh, pose=np.eye(4))

    try:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=frustum_params["width"],
            viewport_height=frustum_params["height"],
        )
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    except Exception as e:
        print(e)
        print("pyrender: Failed rendering frame")
        color = np.zeros(
            (frustum_params["height"], frustum_params["width"], 3), dtype="uint8"
        )

    return color[..., ::-1]


if __name__ == "__main__":
    main()
