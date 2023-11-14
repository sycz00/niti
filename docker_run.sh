#!/bin/sh
NITI_PATH="/media/fa/Shared_Files/Fraunhofer/niti"
GPU=0
#build argument -t name-tag and the point stands for everything
docker build -t wangmaolin/niti:0.1 .

#--shm-size			Size of /dev/shm
#--runtime			Runtime to use for this container
#--env	-e		Set environment variables
#-v Bind mount a volume Easy way to volume mount on docker run command to exchange files between container and host
# -v /path/from/your/host:/path/inside/the/container
#   <-------host------->:<--------container------->
#--interactive	-i		Keep STDIN open even if not attached
#--tty	-t		Allocate a pseudo-TTY
docker run --shm-size=2g --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU -v $NITI_PATH:/niti -it wangmaolin/niti:0.1
