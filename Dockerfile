FROM python:bullseye

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl git unzip && pip install --upgrade pip setuptools
RUN pip install numpy matplotlib
RUN pip cache purge && apt clean && rm -rf /var/lib/apt/lists/*
