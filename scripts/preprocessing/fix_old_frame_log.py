# %%
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

import flexiznam as flz

project="blota_onix_pilote"
session_name="BRYA142.5d_S20230831"

def fix_old_frame_log(folder):
    """Fix old frame loggers that do not contain all columns.
    
    Args:
        folder (Path): path to recording folder.
        
    Returns:
        frame_log (pd.DataFrame): frame log with all columns.
    """
    folder  = Path(folder)
    frame_log = pd.read_csv(folder / "FrameLog.csv")
    photodiode_log = pd.read_csv(folder / "PhotodiodeLog.csv")
    rotary_log = pd.read_csv(folder / "RotaryEncoder.csv")
    frame_log.rename(columns={"Frame": "FrameIndex"}, inplace=True)

    # group loggers by frame index
    photodiode_log = photodiode_log.groupby("Frame").median().reset_index()
    rotary_log = rotary_log.groupby("Frame").median().reset_index()
    # crop to last frame_log
    photodiode_log = photodiode_log[photodiode_log.Frame <= frame_log.FrameIndex.max()]
    rotary_log = rotary_log[rotary_log.Frame <= frame_log.FrameIndex.max()]

    # put relevant columns in frame_log
    frame_log['PhotoQuadColor'] = np.nan
    frame_log.loc[photodiode_log.Frame, 'PhotoQuadColor'] = photodiode_log['PhotoQuadColor'].values
    for w in ['MouseZ', 'EyeZ']:
        frame_log[w] = np.nan
        frame_log.loc[rotary_log.Frame, w] = rotary_log[w].values
    return frame_log

flm_sess = flz.get_flexilims_session(project_id=project)
sess = flz.get_entity(name=session_name, datatype="session", flexilims_session=flm_sess)
visstim = flz.get_datasets_recursively(origin_id=sess.id, dataset_type="visstim", 
                                       flexilims_session=flm_sess)
# %%
for rec, ds_dict in visstim.items():
    for ds in ds_dict:
        fl = fix_old_frame_log(ds.path_full)
        # make a backup if it doesn't exist
        if not (ds.path_full / "FrameLog.csv.backup").exists():
            shutil.copy(ds.path_full / "FrameLog.csv", ds.path_full / "FrameLog.csv.backup")
        fl.to_csv(ds.path_full / "FrameLog.csv", index=False)
# %%
