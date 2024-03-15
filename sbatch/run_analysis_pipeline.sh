#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=47:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage2

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
echo Run depth fit ${RUN_DEPTH_FIT}
echo Run rf fit ${RUN_RF}
echo Run rs of fit ${RUN_RSOF_FIT}
echo Run plot ${RUN_PLOT}
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/pipelines/"
python analysis_pipeline.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${RUN_DEPTH_FIT} ${RUN_RF} ${RUN_RSOF_FIT} ${RUN_PLOT}