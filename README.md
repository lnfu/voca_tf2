# VOCA 使用 TensorFlow 2 重寫

## Training

```sh
docker compose up train
```

### Environment Variables

- `TF_CPP_MIN_LOG_LEVEL=1`

### Python Packages

- `resampy==0.4.3`
- `python-speech-features==0.6`

### Ubuntu Packages

## Inference

```sh
docker compose up run
```

### Environment Variables

- `PYOPENGL_PLATFORM=osmesa`
- `MUJOCO_GL=osmesa`

### Python Packages

- `resampy==0.4.3`
- `python-speech-features==0.6`
- `opencv-python==4.10.0.84`
- `trimesh==4.4.9`
- `pyrender==0.1.45`
- `meshio==5.3.5`
- `pyopengl==3.1.4`
- `networkx==3.4.1`

### Ubuntu Packages

- `libosmesa6`
- `ffmpeg`

## For Developers

```sh
docker compose up dev
```

## 合併影片、音訊

推薦使用 Kdenlive 開源剪輯軟體

也可以使用 ffmpeg

```sh
ffmpeg -i audio.wav -i video.mp4 -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p merged.mp4
```

## 規範

- mesh = vertex + face
- pcd (point cloud) = shape(5023, 3)

mesh to image 使用 pyrender

影片加上音訊使用 ffmpeg

忽略 pyrender 對 pyopengl 版本的要求: https://blog.csdn.net/guntangsanjing/article/details/127651381

pyopengl 必須安裝 3.1.4
