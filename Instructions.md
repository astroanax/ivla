#Instructions

sudo apt update
sudo apt install curl vim tmux -y

## Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

## Post-Processing
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
 
## Clone the repo
git clone --depth 1 --recurse-submodules https://github.com/shehiin/InternManip.git

## Clone the baseimage from 
docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0

## Or Clone the base image we committed into the system
docker pull docker.io/r2d208/internmanip:v1

## Or Clone the 18k trained image we committed from the hub
docker pull docker.io/r2d208/intern_18k:v1

## Get the model weights:
cd InternManip
mkdir data
cd data
mkdir model
git clone https://huggingface.co/InternRobotics/Gr00t-n1-5_Genmanip_IROS model/

## Get the dataset using 

## Check for python
python3 --version

### If no python:
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

## Check for pip
pip3 --version

### If no pip:
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3

## install huggingface_hub
pip3 install huggingface_hub
pip install huggingface_hub

huggingface-cli download InternRobotics/IROS-2025-Challenge-Manip \
  --repo-type dataset \
  --include "train_real/*" \
  --local-dir ./IROS-2025-Challenge-Manip


## Run the container
### Double check if the dataset and the model are present

docker run --name internmanip -it --rm --privileged   --gpus all   --network host   -e "ACCEPT_EULA=Y"   -e "PRIVACY_CONSENT=Y"   -e "DISPLAY=${DISPLAY}"   --entrypoint /bin/bash   -w /root/InternManip   -v /tmp/.X11-unix/:/tmp/.X11-unix   -v ${PWD}:/root/InternManip   -v ${HOME}/data:/root/InternManip/data/dataset   -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw   -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw   -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw   -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw   -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw   -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw   -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw   -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw   crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0

## Inside the container run the following command to train the model

