# VOCA 使用 TensorFlow 2 重寫

```sh
docker run --gpus all -it -v .:/workspace  --rm nvcr.io/nvidia/tensorflow:23.07-tf2-py3 bash
# docker run --gpus all -it -v .:/workspace  --rm nvcr.io/nvidia/tensorflow:24.09-tf2-py3 bash
pip install pyyaml resampy python_speech_features scipy trimesh pyrender
export TF_CPP_MIN_LOG_LEVEL=1

apt install -y ffmpeg 
apt install -y libosmesa6

cd mesh
apt update -y
apt install -y libboost-dev 
# apt install python3-opengl
BOOST_INCLUDE_DIRS=/usr/include/boost make all
cd ..
```

## 規範

- mesh = vertex + face
- pcd (point cloud) = shape(5023, 3)

mesh to image 使用 pyrender

影片加上音訊使用 ffmpeg

<!-- pyrender 需要安裝 mesa (opengl 相關)
```sh
apt update
wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
dpkg -i ./mesa_18.3.3-0.deb || true
apt install -f
apt --fix-broken install
``` -->

忽略 pyrender 對 pyopengl 版本的要求: https://blog.csdn.net/guntangsanjing/article/details/127651381

pyopengl 必須安裝 3.1.4

## 🎯 TODO

- [ ] 研究 cuDNN, cuFFT, and cuBLAS
- [ ] 撰寫 Docker Compose file
- [ ] 自己寫簡單版本的 psbody.mesh (只需要簡單的建立、儲存功能即可)
- [ ] function type hint (單層的全部補上)
