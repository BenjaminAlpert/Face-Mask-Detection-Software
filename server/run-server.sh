#!/bin/bash

#docker run --gpus all -it -p 8888:8888 -v ~/cs254a-final-project/server/jupyter/:/root/.jupyter/ -v ~/cs254a-final-project:/tf tensorflow/tensorflow:latest-gpu-jupyter
docker run --gpus all -it -p 8888:8888 -v ~/cs254a-final-project/server/jupyter/:/root/.jupyter/ -v ~/cs254a-final-project:/tf tensorflow/tensorflow:2.6.0-gpu-jupyter
