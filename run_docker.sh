#!/bin/bash

# Define port and image name
CONTAINER_PORT=8888
IMAGE_NAME="jupyter/base-notebook:python-3.8"

# Pull the image
docker pull $IMAGE_NAME
 
# Run the container
docker run -it --rm -p $CONTAINER_PORT:$CONTAINER_PORT -v $(pwd):/home/jovyan/ $IMAGE_NAME

