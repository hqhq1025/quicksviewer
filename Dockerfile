FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# RUN chsh -s /bin/bash
# SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda/bin:$PATH"


# WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    libjpeg8-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    software-properties-common \
    net-tools \
    lsof \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    ninja-build \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -f -y python3.11 python3.11-venv python3.11-dev python3-pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python
ENV CFLAGS="-I/usr/bin/python3.11"
RUN apt-get install -y python3.11-distutils



# set language
ENV LANG zh_CN.UTF-8
ENV LANGUAGE zh_CN:zh


# # Configure SSH
RUN apt-get update && apt-get install -f -y --no-install-recommends openssh-server whois
RUN echo 'service ssh restart' >> /root/start.sh


RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --upgrade pip --ignore-installed && pip cache purge
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
RUN pip install --trusted-host torch==2.4.1 \
                torchvision \
                decord \
                opencv-python \
                fairscale \
                openai \
                jsonlines \
                deepspeed==0.16.5 \
                pytest \
                matplotlib \
                datasets==3.2.0 \
                seaborn \
                braceexpand \
                numpy==2.2.2 \
                accelerate==1.3.0 \
                transformers==4.44.0 \
                ipdb \
                multiprocess==0.70.16 \
                ninja \
                tensorboard \
                wandb \
                triton==3.1.0

# Install Flash-Attention
RUN wget -P /tmp https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
  && python3 -m pip install /tmp/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN python -c 'import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()'


# CMD ["/bin/bash"]
