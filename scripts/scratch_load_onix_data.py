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

matplotlib.use('MacOSX')
import cv2
import pandas as pd
from cottage_analysis.io_module import harp
from cottage_analysis.io_module import onix
from cottage_analysis.io_module.video.io_func import deinterleave_camera
from depth_analysis_2p.vis_stim import format_loggers
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path('/Volumes/lab-znamenskiyp/')
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220817'
EPHYS_REC = 'R171408'
VIS_STIM_REC = 'R165353_SpheresPermTubeReward'

# ROOT = Path('/camp/lab/znamenskiyp/')
DATA_PATH = ROOT / 'data/instruments/raw_data/projects'
DATA_PATH = DATA_PATH / 'blota_onix_pilote' / MOUSE / SESSION
PROCESS_PATH = ROOT / 'home/shared/projects'
PROCESS_PATH = PROCESS_PATH / 'blota_onix_pilote' / MOUSE / SESSION

# load ephys data
ephys_data = onix.load_rhd2164(DATA_PATH / EPHYS_REC)
# same for lighthouse photodiove
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

# Load stimulation data

param_file = DATA_PATH / VIS_STIM_REC / ('%s_%s_%s_NewParams.csv' % (MOUSE, SESSION,
                                                                       VIS_STIM_REC))
vis_stim = pd.read_csv(param_file)
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
# plot what I did
fig = plt.figure()
fig.subplots_adjust(wspace=0.3, hspace=0.4, top=0.98, right=0.98)
ax = fig.add_subplot(2, 2, 1)
ax.plot(harp_onix_clock, np.ones_like(harp_onix_clock), '|', label='Good triggers')
ax.plot(harp_message['onix_clock'], np.zeros_like(harp_message['onix_clock']), '|',
        label='All triggers')
ax.axvline(cutoff, color='k')
ax.set_xlabel('Harp time')
ax.legend(loc=0)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 2)
ax.clear()
ax.hist(np.diff(harp_onix_clock)*1000)
ax.set_xlabel('Intertrigger time (ms)')

ax = fig.add_subplot(2, 2, 3)
ax.clear()
ax.plot(harp_onix_clock, real_time, '.')
ax.set_xlabel('Harp time')
ax.set_ylabel('Onix time')

ax = fig.add_subplot(2, 2, 4)
ax.clear()
ax.plot(real_time, (harp2onix(harp_onix_clock) - real_time) * 1000, 'o')
ax.set_xlabel('Onix time')
ax.set_ylabel('Error (ms)')

# Load spike sorted data
kilosort_folder = PROCESS_PATH / EPHYS_REC / 'kilosort'
ks_data = dict()
for w in ['times', 'clusters']:
    ks_data[w] = np.load(kilosort_folder / ('spike_%s.npy' % w)).reshape(-1)
for w in ['group', 'info']:
    ks_data[w] = pd.read_csv(kilosort_folder / ('cluster_%s.tsv' % w), sep='\t')

# get an example unit, the good with the biggest amplitude
good = ks_data['info'][ks_data['info'].group == 'good']
cluster_id = good.cluster_id[good.amp.idxmax()]
# get unit spike index in ephys
spike_index = ks_data['times'][ks_data['clusters'] == cluster_id]
clock = ephys_data['clock'].reshape(-1)
spike_time = clock[spike_index]
