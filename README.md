# VOCA 使用 TensorFlow 2 重寫

```sh
docker run --gpus all -it -v .:/workspace  --rm nvcr.io/nvidia/tensorflow:23.07-tf2-py3 bash
# docker run --gpus all -it -v .:/workspace  --rm nvcr.io/nvidia/tensorflow:24.09-tf2-py3 bash
pip install pyyaml resampy python_speech_features scipy
export TF_CPP_MIN_LOG_LEVEL=1

cd mesh
apt update
apt install libboost-dev python3-opengl
BOOST_INCLUDE_DIRS=/usr/include/boost make all
```

## 規範

- mesh = vertex + face
- pcd (point cloud) = shape(5023, 3)


## 🎯 TODO

- [ ] 研究 cuDNN, cuFFT, and cuBLAS
- [ ] 撰寫 Docker Compose file
- [ ] 自己寫簡單版本的 psbody.mesh (只需要簡單的建立、儲存功能即可)