import numpy as np

import cv2
import logging
import trimesh
import pyrender
import meshio
import copy


class MeshRenderer:
    def __init__(self) -> None:
        pass

    def render_mesh_to_image(self, mesh: meshio.Mesh, center=np.zeros(3), rotation=np.zeros(3)):

        print(center)

        camera_params = {
            "optical_center": [400.0, 400.0],
            "focal_length": [4754.97941935 / 2, 4754.97941935 / 2],  # TODO 數值?
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
        camera_pose = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]  # 往 z 平移一單位
        scene.add(camera, pose=camera_pose)

        # 加入光源
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.5)
        light_pose = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]  # 往 z 平移一單位
        scene.add(light, pose=light_pose)

        points_copy = np.copy(mesh.points)
        cells_copy = copy.deepcopy(mesh.cells)
        mesh_copy = meshio.Mesh(points=points_copy, cells=cells_copy)

        # 圍繞 center 做旋轉 rotation
        mesh_copy.points[:] = (
            cv2.Rodrigues(rotation)[0].dot((mesh_copy.points - center).T).T + center
        )

        # trimesh 上色 (沒有色彩)
        triangles = np.vstack(
            [cell.data for cell in mesh_copy.cells if cell.type == "triangle"]
        )

        tri_mesh = trimesh.Trimesh(
            vertices=mesh_copy.points, faces=triangles, vertex_colors=None
        )

        # pyrender mesh
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        scene.add(render_mesh, pose=np.eye(4))

        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=frustum_params["width"],
                viewport_height=frustum_params["height"],
            )
            colors, _ = renderer.render(
                scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES
            )
        except Exception as e:
            logging.error("pyrender: An error occurred while rendering: %s", e)
            colors = np.zeros(
                (frustum_params["height"], frustum_params["width"], 3), dtype="uint8"
            )

        return colors[..., ::-1]
