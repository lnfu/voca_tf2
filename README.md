# VOCA 使用 TensorFlow 2 重寫

## Training

```sh
docker compose up train
```

### Environment Variables

- `TF_CPP_MIN_LOG_LEVEL=1`

### Python Packages

- `resampy`
- `python_speech_features`

### Ubuntu Packages

## Inference

```sh
docker compose up run
```

### Environment Variables

- `PYOPENGL_PLATFORM=osmesa`
- `MUJOCO_GL=osmesa`

### Python Packages

- `resampy`
- `python_speech_features`
- `opencv-python`
- `trimesh`
- `pyrender`
- `meshio`
- `pyopengl==3.1.4`

### Ubuntu Packages

- `libosmesa6`
- `ffmpeg`

## For Developers

```sh
docker compose up dev
```

## 規範

- mesh = vertex + face
- pcd (point cloud) = shape(5023, 3)

mesh to image 使用 pyrender

影片加上音訊使用 ffmpeg

忽略 pyrender 對 pyopengl 版本的要求: https://blog.csdn.net/guntangsanjing/article/details/127651381

pyopengl 必須安裝 3.1.4

## 🎯 TODO

- [ ] 研究 cuDNN, cuFFT, and cuBLAS
- [ ] 撰寫 Docker Compose file
- [ ] 自己寫簡單版本的 psbody.mesh (只需要簡單的建立、儲存功能即可)
- [ ] function type hint (單層的全部補上)
