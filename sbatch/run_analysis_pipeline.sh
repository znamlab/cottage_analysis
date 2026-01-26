#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cypranc@crick.ac.uk

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_seq

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
cd "/camp/lab/znamenskiyp/home/users/cypranc/cottage_analysis/cottage_analysis/pipelines/"
python analysis_pipeline.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${RUN_DEPTH_FIT} ${RUN_RF} ${RUN_RSOF_FIT} ${RUN_PLOT}

