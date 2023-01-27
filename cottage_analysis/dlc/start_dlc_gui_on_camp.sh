#!/bin/bash
#
# You first need to connect to a vis node
# srun -p vis -n 1 --gres=gpu:1 -t 00:30:00 --pty  bash

ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130 Anaconda3
ml Tigervnc fluxbox
conda activate DLC
vncstart
