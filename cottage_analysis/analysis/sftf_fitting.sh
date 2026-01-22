#!/bin/bash
#SBATCH --job-name=sftf_fitting          
#SBATCH --mail-type=BEGIN,END,FAIL         
#SBATCH --mail-user=toksozi@crick.ac.uk  
#SBATCH --ntasks=1                                  
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G           
#SBATCH --output=logs/sftf_%j.log    
#SBATCH --partition=ncpu              

# 1. LOAD ENVIRONMENT
module load Anaconda3
source activate cottage

# 2. CHECK FOR VARIABLES
: "${PROJECT:?Error: PROJECT variable is not set.}"
: "${MOUSE:?Error: MOUSE variable is not set.}"
: "${SESSION:?Error: SESSION variable is not set.}"

# Defaults
: "${PROTOCOL:=SFTF}" 
# This is where it looks for RAW data (Input)
: "${BASE_DIR:=/camp/lab/znamenskiyp/home/shared/projects/}"

print_header() {
    echo "========================================"
    echo "JOB STARTED"
    echo "----------------------------------------"
    echo "Project:  $PROJECT"
    echo "Mouse:    $MOUSE"
    echo "Session:  $SESSION"
    echo "Protocol: $PROTOCOL"
    echo "========================================"
}
print_header

# 3. RUN THE PYTHON SCRIPT
# We use the variables ($MOUSE, etc.) that were passed in
python run_sftf_fitting.py \
    --project "$PROJECT" \
    --mouse "$MOUSE" \
    --session "$SESSION" \
    --protocol "$PROTOCOL" \
    --base_dir "$BASE_DIR" \
    --niter 50   # High iterations since we are on the cluster

echo "Job finished."