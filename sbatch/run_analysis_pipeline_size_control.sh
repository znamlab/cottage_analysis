#!/bin/bash
#
#SBATCH --job-name=size
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage2

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/pipelines/"
python analysis_pipeline_size_control.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} 