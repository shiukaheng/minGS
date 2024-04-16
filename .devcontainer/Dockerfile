FROM ubuntu:focal

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Setup essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    git \
    curl \
    wget \
    unzip \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# CA certificates
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit 12
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-4 && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install basic python packages
RUN pip3 install --upgrade pip && \
    pip3 install \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    pandas \
    jupyterlab \
    ipywidgets \
    ipykernel

# Install pytorch
RUN pip3 install torch torchvision torchaudio

# Install opencv
RUN pip3 install opencv-python

# Install misc
RUN pip3 install plyfile lpips pybind11 viser