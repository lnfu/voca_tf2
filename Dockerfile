FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3 
# or switch to 24.09-tf2-py3 if needed

# ENV TF_CPP_MIN_LOG_LEVEL 1

ARG USERNAME=efliao
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYOPENGL_PLATFORM osmesa
ENV MUJOCO_GL osmesa
ENV TF_ENABLE_ONEDNN_OPTS 0

RUN apt-get update && apt-get install -y \
    make \
    ffmpeg \
    git \
    libosmesa6 \
    python-is-python3 \
    sudo \
    tini \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN echo $USERNAME

RUN mkdir /app \
    && chown $USERNAME:$USERNAME /app

USER $USERNAME
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    autopep8==2.3.1 \
    flake8==7.1.1 \
    pycodestyle==2.12.1

RUN pip install \
    resampy==0.4.3 \
    python-speech-features==0.6 \
    opencv-python==4.10.0.84 \
    trimesh==4.4.9 \
    pyrender==0.1.45 \
    meshio==5.3.5 \
    networkx==3.4.1 \ 
    chumpy==0.70 

RUN pip install pyopengl==3.1.4

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["tail", "-f" ,"/dev/null"]
