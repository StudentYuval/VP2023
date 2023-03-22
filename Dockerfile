FROM ubuntu:18.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl bzip2 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -sS -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/miniconda.sh

# Set environment variables
ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=myenv
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV

# Create and activate a new environment
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean --all --yes
ENV PATH="$CONDA_PREFIX/bin:${PATH}"
