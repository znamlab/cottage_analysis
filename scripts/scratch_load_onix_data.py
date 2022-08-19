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
import pandas as pd

matplotlib.use('MacOSX')
from cottage_analysis.io_module import harp
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path('/Volumes/lab-znamenskiyp/')
# ROOT = Path('/camp/lab/znamenskiyp/')
DATA_PATH = ROOT / 'data/instruments/raw_data/projects'
DATA_PATH = DATA_PATH / 'blota_onix_pilote/BRAC6692.4a/S20220817/'
PROCESS_PATH = ROOT / 'home/shared/projects'
PROCESS_PATH = PROCESS_PATH / 'blota_onix_pilote/BRAC6692.4a/S20220817/'


# Harp
behaviour_folder = DATA_PATH / 'R165353_SpheresPermTubeReward'
bin_file = behaviour_folder / 'BRAC6692.4a_S20220817_R165353_SpheresPermTubeReward_harpmessage.bin'
harp_message = harp.read_message(path_to_file=bin_file)

# harp contains a lot of stuff, split that
# I'll make a dataframe to make it easier, it takes a while
print('Making dataframe')
harp_message = pd.DataFrame(harp_message)
print('Done')
# Each message has a message type that can be 'READ', 'WRITE', 'EVENT', 'READ_ERROR',
# or 'WRITE_ERROR'.
# We don't want error
msg_types = harp_message.msg_type.unique()
assert not np.any([m.endswith('ERROR') for m in msg_types])
# READ events are the initial config loading at startup. We don't care
harp_message = harp_message[harp_message.msg_type != 'READ']

# WRITE messages are mostly the rewards.
# The reward port is toggled by writing to register 36, let's focus on those events
reward_message = harp_message[harp_message.address == 36]
harp_reward_times = reward_message.timestamp_s.values

# EVENT messages are analog and digital input.
# Analog are the photodiode and the rotary encoder, both on address 44
analog = harp_message[harp_message.address == 44]
harp_analog_times = analog.timestamp_s.values
analog = np.vstack(analog.data)
harp_frame_photodiode = analog[:, 0]
harp_rotary_encoder = analog[:, 1]

# Digital input is on address 32, the data is 2 when the trigger is high
di = harp_message[(harp_message.address == 32) & (harp_message.data == (2,))]
harp_onix_trigger_times = di.timestamp_s.values

# Ephys ephys.
# This should be a simple binary file with uint16. Don't forget the U
ephys = DATA_PATH / 'R171408' / r'rhd2164-ephys_2022-08-17T17_14_08.raw'
n_pts = ephys.stat().st_size / 2  # divide by 2 for uint16
ephys_data = np.memmap(ephys, dtype='uint16', mode='r', order='F',
                       shape=(64, int(n_pts/64)))

# plot the data
time_window = 0.2
fs = 30000
v_shift = 400
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
data2plot = (np.array(ephys_data[:4, :int(fs*time_window)], float) - 2**15) * 0.195
ax.plot(np.arange(int(fs*time_window))/fs,  data2plot.T + np.arange(4) * v_shift, lw=0.4)
ax.set_title('Ephys')

ax = fig.add_subplot(2, 1, 2)
ax.set_title('Harp')
ax.clear()
ht0 = harp_onix_trigger_times[0]  # 0 the hard at the first onix trigger ... but here
# it does not quite work as the triggering was started, then stopped and restarted

ax.plot(harp_analog_times - ht0, harp_frame_photodiode/800)
ax.plot(harp_analog_times - ht0, harp_rotary_encoder/32000)
ax.plot(harp_reward_times - ht0, np.zeros_like(harp_reward_times), 'o')
ax.plot(harp_onix_trigger_times - ht0, np.zeros_like(harp_onix_trigger_times) - 1, '|')