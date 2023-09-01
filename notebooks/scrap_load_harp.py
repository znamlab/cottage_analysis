from pathlib import Path
import cottage_analysis.io_module.onix as onix
import cottage_analysis.io_module.harp as harp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 

processed_path = Path('/camp/lab/znamenskiyp/home/shared/projects/blota_onix_pilote/')
data_path = Path('/camp/lab/znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote/')
mouse = 'BRAC7448.2d'
session = 'S20230404'
session_path = data_path / mouse / session

processed_dir = processed_path/mouse/session
processed_dir.mkdir(exist_ok=True)

ephys = 'R110119'
stimulus = 'R110246_SpheresPermTubeReward'
headfixed_cam = '110357' 

ephys_path = session_path / ephys
stimulus_path = session_path / stimulus
headfixed_cam_path = session_path / headfixed_cam

processed_messages = processed_path / stimulus / 'processed_harp.npz'

processed_messages.parent.mkdir(exist_ok=True)

if Path(processed_messages).is_file():
    harp_message = dict(np.load(processed_messages))
else:
    # slow part: read harp messages so save output and reload
    harp_message = onix.load_harp(stimulus_path / 'harpmessage.bin')
    np.savez(processed_messages, **harp_message)