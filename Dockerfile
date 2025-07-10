FROM ubuntu:22.04

# System dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    tk-dev libffi-dev uuid-dev git curl

# Install Python 3.12 from source
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar xzf Python-3.12.2.tgz && \
    cd Python-3.12.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Set Python 3.12 as default
RUN ln -s /usr/local/bin/python3.12 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install torch pytorch-ignite imageio tqdm opencv-python nibabel "protobuf<=3.20.3" matplotlib matplotlib-inline
RUN pip install monai sagemaker pandas

# Set working directory
WORKDIR /opt/ml/code

# Copy training code
COPY . .

# Default command
ENTRYPOINT ["python", "train_gpt.py"]
