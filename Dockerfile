# Docker image:
# following properties are:
# 1. based on ubuntu 18.04
# 2. set everything to work with conda environment base on environment.yml in this folder
# 3. install all the dependencies for the project
# 4. activate the conda environment

FROM ubuntu:18.04

# set platform to platform=linux/x86_64
ENV PLATFORM=linux/x86_64

# set the working directory in the container
WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    mercurial \
    subversion \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 


# install miniconda virtual environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

COPY . .