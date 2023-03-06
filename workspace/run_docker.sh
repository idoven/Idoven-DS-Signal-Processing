#!/bin/bash
docker run --gpus all -p 8888:8888 --mount type=bind,src=$(pwd),dst=/tf \
  -it --rm tensorflow/tensorflow:nightly-gpu-jupyter jupyter notebook \
  --port 8888 --ip 0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''
