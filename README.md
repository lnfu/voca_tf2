# VOCA 使用 TensorFlow 2 重寫

```sh
docker run --gpus all -it -v .:/workspace  --rm nvcr.io/nvidia/tensorflow:24.09-tf2-py3 bash
pip install tensorflow pyyaml resampy python_speech_features
export TF_CPP_MIN_LOG_LEVEL=1
```

## 🎯 TODO

- [ ] 研究 cuDNN, cuFFT, and cuBLAS
- [ ] 撰寫 Docker Compose file
