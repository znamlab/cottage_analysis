#!/bin/bash
#
#SBATCH --job-name=2p-preprocess-gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/logs/2p_analysis_%j.log"
ml purge
ml CUDA/11.1.1-GCC-10.2.0
ml cuDNN/8.0.5.39-CUDA-11.1.1
ml Anaconda3/2020.07

source activate base

conda activate 2p_analysis_cottage

echo Processing ${SESSION} from ${MOUSE} in project ${PROJECT}...
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/depth_analysis/depth_preprocess/"
python find_depth_neurons.py ${PROJECT} ${MOUSE} ${SESSION}