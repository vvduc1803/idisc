FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt update

RUN apt-get clean -y

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install -y python3.9 python3-pip sudo

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update

RUN useradd -m ana

RUN chown -R ana:ana /home/ana

COPY --chown=ana . /home/ana/Study/CVPR/idisc

USER ana

RUN cd /home/ana/Study/CVPR/idisc && pip3 install -r requirements.txt

WORKDIR /home/ana/Study/CVPR/idisc