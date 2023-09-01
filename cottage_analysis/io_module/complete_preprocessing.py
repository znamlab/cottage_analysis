"""
Example script to load ONIX data

Should load:
- harp data
- video timestamps
- ephys data
- BNO data
- TS data
"""
import matplotlib

from cottage_analysis.imaging.common import find_frames
from cottage_analysis.imaging.common.find_frames import find_pulses

#matplotlib.use('MacOSX')
import cv2
import pandas as pd
from cottage_analysis.io_module import harp
from cottage_analysis.io_module import onix
from cottage_analysis.io_module.video.io_func import deinterleave_camera
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path('/camp/lab/znamenskiyp/')
MOUSE = 'BRAC7448.2d'
SESSION = 'S20230330'
EPHYS_REC = 'R115005'
VIS_STIM_REC = 'R115126_SpheresPermTubeReward'

# ROOT = Path('/camp/lab/znamenskiyp/')
DATA_PATH = ROOT / 'data/instruments/raw_data/projects'
DATA_PATH = DATA_PATH / 'blota_onix_pilote' / MOUSE / SESSION
PROCESS_PATH = ROOT / 'home/shared/projects'
PROCESS_PATH = PROCESS_PATH / 'blota_onix_pilote' / MOUSE / SESSION

PROCESS_PATH.mkdir(exist_ok=True)

# load ephys data
onix_sampling = 250e6
ephys_data = onix.load_rhd2164(DATA_PATH / EPHYS_REC)
# same for lighthouse photodiode
ts_data = onix.load_ts4231(DATA_PATH / EPHYS_REC)

# Load harp messages
processed_messages = PROCESS_PATH / VIS_STIM_REC / 'processed_harp.npz'
processed_messages.parent.mkdir(exist_ok=True)
if Path(processed_messages).is_file():
    harp_message = dict(np.load(processed_messages))
else:
    # slow part: read harp messages so save output and reload
    harp_message = onix.load_harp(DATA_PATH / VIS_STIM_REC / ('%s_%s_%s_harpmessage.bin'
                                  % (MOUSE, SESSION, VIS_STIM_REC)))
    np.savez(processed_messages, **harp_message)
# make a speed out of rotary increament
mvt = np.diff(harp_message['rotary'])
rollover = np.abs(mvt > 40000)
mvt[rollover] -= 2**16 * np.sign(mvt[rollover])
# The rotary count decreases when the mouse goes forward
mvt *= -1
harp_message['mouse_speed'] = np.hstack([0, mvt])  # 0-padding to keep constant length

# Load stimulation data

#param_file = DATA_PATH / VIS_STIM_REC / ('%s_%s_%s_NewParams.csv' % (MOUSE, SESSION,
#                                                                       VIS_STIM_REC))
#vis_stim = pd.read_csv(param_file)
# add onix time to vis_stim
# find triggers from harp
# there was a double acquisition, get read of early trigger
cutoff = 12547030
harp_onix_clock = harp_message['onix_clock']
harp_onix_clock = np.array(harp_onix_clock[harp_onix_clock > cutoff])
real_time = np.arange(len(harp_onix_clock)) * 10e-3  # assume a perfect 100Hz
# linear regression:
t0 = harp_onix_clock[0]
slope = np.nanmean(real_time / (harp_onix_clock - t0))


def harp2onix(data):
    """Convert harp timestamp in onix time"""
    return (data - t0) * slope


vis_stim['onix_time'] = harp2onix(vis_stim['HarpTime'])
harp_message['analog_onix_time'] = harp2onix(harp_message['analog_time'])