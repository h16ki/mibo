FROM python:3.9-bullseye

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    build-essential

RUN pip install --upgrade pip setuptools
RUN pip install torch==1.13.1
RUN pip install torchvision==0.10.0 -f https://download.pytorch.org/whl/torchvision/
RUN pip install numpy matplotlib scipy pandas opencv-python tqdm
RUN pip cache purge && apt clean && rm -rf /var/lib/apt/lists/*
