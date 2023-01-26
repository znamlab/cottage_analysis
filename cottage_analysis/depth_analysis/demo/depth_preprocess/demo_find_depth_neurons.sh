#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/depth_analysis/logs/2p_analysis_%j.log"

ml purge

ml Anaconda3/2020.07
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate 2p_analysis_clone

echo Processing ${SESSION} ${PROTOCOL} from ${MOUSE} in project ${PROJECT}...
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/depth_analysis/depth_preprocess/"
python find_depth_neurons.py ${PROJECT} ${MOUSE} ${SESSION} ${PROTOCOL}