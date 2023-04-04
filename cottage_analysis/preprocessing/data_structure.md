# Logger dataframes
## 1. vs_df
    - columns: monitor_frame, harp_time, imaging_frame, depth, mouse_z, eye_z, spheres_no(list), spheres_start_theta(list, startTheta), spheres_start_z(list, startZ), spheres_x, spheres_y, spheres_z, closed_loop(bool)
    - made from: MonitorFrame_df, ImageTriggers_df, ParamLog_df
    - take the params/runningZ right before the monitor_frame time
## 2. trials_df
    - columns: trial_no, trial_stim_start_time, trial_stim_stop_time, trial_blank_start_time, trial_blank_stop_time, depth, stim_start_imaging_frame, stim_stop_imaging_frame, blank_start_imaging_frame, blank_stop_imaging_frame, RS_array_stim (1xframes), RS_array_blank (1xframes), OF_array_stim (1xframes), OF_array_blank (1xframes), spheres_no(list), closed_loop(bool)
## 3. img_df
    - columns: image_frame, monitor_frame, harp_time, trial_no, closed_loop(bool), depth, mouse_z, eye_z, of 