#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map

echo Processing ${SESSION_NAME} in project ${PROJECT}...
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/pipelines/"
python depth_SFTF.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL}