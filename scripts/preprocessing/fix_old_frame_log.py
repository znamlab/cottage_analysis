"""
This script was written for old ephys files in which the frame log was split into
several files. It should not be needed anymore.
"""
# %%
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

import flexiznam as flz

project = "blota_onix_pilote"
session_name = "BRYA142.5d_S20230831"
session_name = "BRAC6692.4a_S20220831"
"""
PROJECT = "blota_onix_pilote"
MOUSE = "BRAC6692.4a"
SESSION = "S20220831"
ONIX_RECORDING = "R163359"
VIS_STIM_RECORDING = "R163332_SpheresPermTubeReward"

"""


def fix_old_frame_log(folder, csv_files=None):
    """Fix old frame loggers that do not contain all columns.

    Args:
        folder (Path): path to recording folder.

    Returns:
        frame_log (pd.DataFrame): frame log with all columns.
    """
    if csv_files is None:
        csv_files = dict(
            FrameLog="FrameLog.csv",
            PhotodiodeLog="PhotodiodeLog.csv",
            RotaryEncoder="RotaryEncoder.csv",
            NewParams="NewParams.csv",
        )
    folder = Path(folder)
    frame_log = pd.read_csv(folder / csv_files["FrameLog"])
    photodiode_log = pd.read_csv(folder / csv_files["PhotodiodeLog"])
    rotary_log = pd.read_csv(folder / csv_files["RotaryEncoder"])
    frame_log.rename(columns={"Frame": "FrameIndex"}, inplace=True)

    # group loggers by frame index
    photodiode_log = photodiode_log.groupby("Frame").median().reset_index()
    rotary_log = rotary_log.groupby("Frame").median().reset_index()
    # crop to last frame_log
    photodiode_log = photodiode_log[photodiode_log.Frame <= frame_log.FrameIndex.max()]
    rotary_log = rotary_log[rotary_log.Frame <= frame_log.FrameIndex.max()]

    # put relevant columns in frame_log
    frame_log["PhotoQuadColor"] = 0
    frame_log.loc[photodiode_log.Frame, "PhotoQuadColor"] = photodiode_log[
        "PhotoQuadColor"
    ].values
    for w in ["MouseZ", "EyeZ"]:
        frame_log[w] = np.nan
        frame_log.loc[rotary_log.Frame, w] = rotary_log[w].values
        # interpolate missing values
        frame_log[w] = frame_log[w].interpolate()
    return frame_log, photodiode_log, rotary_log


flm_sess = flz.get_flexilims_session(project_id=project)
sess = flz.get_entity(name=session_name, datatype="session", flexilims_session=flm_sess)
visstim = flz.get_datasets_recursively(
    origin_id=sess.id, dataset_type="visstim", flexilims_session=flm_sess
)
harpds = flz.get_datasets_recursively(
    origin_id=sess.id, dataset_type="harp", flexilims_session=flm_sess
)
# %%
# for rec, ds_dict in visstim.items():
for rec, ds_dict in harpds.items():
    for ds in ds_dict:
        if "R163332_SpheresPermTubeReward" not in ds.dataset_name:
            continue
        csv = ds.extra_attributes["csv_files"]
        frame_log = pd.read_csv(ds.path_full / csv["FrameLog"])
        photodiode_log = pd.read_csv(ds.path_full / csv["PhotodiodeLog"])
        rotary_log = pd.read_csv(ds.path_full / csv["RotaryEncoder"])

        fl, pl, rl = fix_old_frame_log(ds.path_full, csv_files=csv)
        # make a backup if it doesn't exist
        backupfile = ds.path_full / (csv["FrameLog"] + ".backup")
        if not backupfile.exists():
            shutil.copy(ds.path_full / csv["FrameLog"], backupfile)
        try:
            fl.to_csv(ds.path_full / csv["FrameLog"], index=False)
        except PermissionError:
            print(f"Could not write to {ds.path_full / csv['FrameLog']}")
            processed = flz.get_data_root("processed", flexilims_session=flm_sess)
            raw = flz.get_data_root("raw", flexilims_session=flm_sess)
            target = processed / ds.path_full.relative_to(raw) / csv["FrameLog"]
            fl.to_csv(target, index=False)

        param_log = pd.read_csv(ds.path_full / csv["NewParams"])
        backupfile = ds.path_full / (csv["NewParams"] + ".backup")
        if not backupfile.exists():
            shutil.copy(ds.path_full / csv["NewParams"], backupfile)
        param_log.rename(columns={"Frame": "Frameindex"}, inplace=True)
        param_log.to_csv(ds.path_full / csv["NewParams"], index=False)
# %%
