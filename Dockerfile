FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3 
# or switch to 24.09-tf2-py3 if needed

# ENV TF_CPP_MIN_LOG_LEVEL 1

ENV PYOPENGL_PLATFORM osmesa
ENV MUJOCO_GL osmesa

RUN apt-get update && apt-get install -y sudo vim libosmesa6 ffmpeg tini

RUN pip install resampy==0.4.3 \ 
    python-speech-features==0.6 \
    opencv-python==4.10.0.84 \
    trimesh==4.4.9 \
    pyrender==0.1.45 \
    meshio==5.3.5 \
    networkx==3.4.1 \ 
    chumpy==0.70 

RUN pip install pyopengl==3.1.4


RUN groupadd -g 1000 efliao

RUN useradd --uid 1000 --gid 1000 --groups root,sudo,adm,users --create-home --password "`openssl passwd -6 -salt XX 12345678`" --shell /bin/bash efliao
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/efliao
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["tail", "-f" ,"/dev/null"]
