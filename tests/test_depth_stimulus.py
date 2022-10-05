from pathlib import Path
import flexiznam as flm
import numpy as np
from matplotlib import pyplot as plt

from cottage_analysis import ephys
from cottage_analysis.ephys.preprocessing import find_frame_start_hf
from cottage_analysis.io_module import onix
from cottage_analysis.stimulus_structure import spheres_tube


PROJECT = 'blota_onix_pilote'
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220831'
ONIX_RECORDING = 'R163359'
VIS_STIM_RECORDING = 'R163332_SpheresPermTubeReward'

# get frame times
data_root = flm.PARAMETERS['data_root']
raw = Path(data_root['raw']) / PROJECT / MOUSE / SESSION
processed = Path(data_root['processed']) / PROJECT / MOUSE / SESSION

data = onix.load_onix_recording(PROJECT, MOUSE, SESSION,
                                vis_stim_recording=VIS_STIM_RECORDING,
                                onix_recording=ONIX_RECORDING, allow_reload=True)
data = ephys.preprocessing.preprocess_exp(data, plot_dir=processed / VIS_STIM_RECORDING)

frame_times = data['hf_frames']
params_log = data['vis_stim_log']['NewParams']
photodiode_log = data['vis_stim_log']['PhotodiodeLog']
frame_log = data['vis_stim_log']['FrameLog']


s = 35.9
e = 36.1

pd_ai_time = data['harp_message']['analog_time_onix']
pd_ai = data['harp_message']['photodiode']
pd_ai_zsc = (pd_ai - pd_ai.mean()) / pd_ai.std()

valid = np.logical_and(pd_ai_time > s, pd_ai_time < e)
plt.plot(pd_ai_time[valid], pd_ai_zsc[valid], label='Photodiode')
valid = np.logical_and(frame_times.onix_time > s, frame_times.onix_time < e)
for t in frame_times.onix_time[valid]:
    plt.axvline(t , color='k')
valid = np.logical_and(frame_log.onix_time > s, frame_log.onix_time < e)
plt.plot(frame_log.onix_time[valid], np.ones(np.sum(valid)), 'o',
         label='Bonsai frames')
plt.legend(loc='lower left')
plt.xlabel('Time (s)')
plt.show()

plt.hist(np.diff(frame_times.onix_time.values)*1000, bins=np.arange(200),
         histtype='step', label='Photodiode')
plt.hist(np.diff(frame_log.onix_time.values)*1000, bins=np.arange(200), histtype='step',
         label='Bonsai')
plt.yscale('log')
plt.legend(loc='upper right')
plt.xlabel('Interframe interval (ms)')
plt.show()