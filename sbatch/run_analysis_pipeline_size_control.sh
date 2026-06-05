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

conda activate v1_depth_map
if [ -z ${ANATOMICAL_ONLY+x} ]; then
    ANATOMICAL_ONLY="True"
fi

if [ -z ${AST_NEUROPIL+x} ]; then
    AST_NEUROPIL="False"
fi
if [ -z ${USE_ANNOTATED+x} ]; then
    USE_ANNOTATED="False"
fi

echo Use anatomical only datasets: ${ANATOMICAL_ONLY}
echo Use ASt neuropil correction: ${AST_NEUROPIL}
echo Use annotated dataset: ${USE_ANNOTATED}
cd "/camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis/cottage_analysis/pipelines/"
python analysis_pipeline_size_control.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} \
    $( [ "${ANATOMICAL_ONLY}" = "False" ] && echo "--no-anatomical-only" ) \
    $( [ "${AST_NEUROPIL}" = "True" ] && echo "--ast-neuropil" ) \
    $( [ "${USE_ANNOTATED}" = "True" ] && echo "--use-annotated" )
