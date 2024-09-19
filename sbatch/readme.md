To run a script: sbatch --export=PROJECT="hey2_3d-vision_foodres_20220101",MOUSE={MOUSE_NAME},SESSION={SESSION_NAME} run_xxx.sh
To check the log: tail -f ../logs/2p_analysis_{JOB_ID}.log

for example: sbatch --export=PROJECT="hey2_3d-vision_foodres_20220101",SESSION_NAME=PZAH8.2f_S20230126,CONFLICTS=skip,PHOTODIODE_PROTOCOL=5 sbatch/run_analysis_pipeline.sh 