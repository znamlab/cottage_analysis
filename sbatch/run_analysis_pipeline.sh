#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
# set defaut values for optional arguments
if [ -z ${CONFLICTS+x} ]; then
    CONFLICTS="skip"
fi
if [ -z ${USE_SLURM+x} ]; then
    USE_SLURM="True"
fi
if [ -z ${RUN_DEPTH_FIT+x} ]; then
    RUN_DEPTH_FIT="True"
fi
if [ -z ${RUN_RF+x} ]; then
    RUN_RF="True"
fi
if [ -z ${RUN_RSOF_FIT+x} ]; then
    RUN_RSOF_FIT="True"
fi
if [ -z ${RUN_PLOT+x} ]; then
    RUN_PLOT="True"
fi
echo Run depth fit ${RUN_DEPTH_FIT}
echo Run rf fit ${RUN_RF}
echo Run rs of fit ${RUN_RSOF_FIT}
echo Run plot ${RUN_PLOT}
# If PROTOCOL_BASE is not set, set it to the default value
if [ -z ${PROTOCOL_BASE+x} ]; then
    PROTOCOL_BASE="SpheresPermTubeReward"
fi
cd "/camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis/cottage_analysis/pipelines/"
python analysis_pipeline.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${RUN_DEPTH_FIT} ${RUN_RF} ${RUN_RSOF_FIT} ${RUN_PLOT} ${PROTOCOL_BASE}
