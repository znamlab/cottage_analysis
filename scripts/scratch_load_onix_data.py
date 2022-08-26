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
onix_sampling = 250e6
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
# make a speed out of rotary increament
mvt = np.diff(harp_message['rotary'])
rollover = np.abs(mvt > 40000)
mvt[rollover] -= 2**16 * np.sign(mvt[rollover])
# The rotary count decreases when the mouse goes forward
mvt *= -1
harp_message['mouse_speed'] = np.hstack([0, mvt])  # 0-padding to keep constant length

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
harp_message['analog_onix_time'] = harp2onix(harp_message['analog_time'])
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

# get good units
good = ks_data['info'][ks_data['info'].group == 'good']
clock = np.array(ephys_data['clock'][0])
good_units = {}
for cluster_id in good.cluster_id.values:
    # get unit spike index in ephys
    spike_index = ks_data['times'][ks_data['clusters'] == cluster_id]
    spike_time = clock[spike_index] / onix_sampling
    good_units[cluster_id] = spike_time

# make a plot of response per depth
# Find corridor change, we start in the first corridor, then go to -9999 and then to
# the next corridor
depth_trace = vis_stim.Depth.values
corridor_starts, = np.where(np.hstack([True, np.diff(depth_trace) > 4000]))
corridor_ends, = np.where(np.hstack([False, np.diff(depth_trace) < -4000]))
# make a simpler dataframe
corridor_df = []
for s, e in zip(corridor_starts, corridor_ends):
    corridor_df.append(dict(start_index=s, start_time=vis_stim.onix_time.iloc[s],
                            end_index=e, end_time=vis_stim.onix_time.iloc[e],
                            depth=vis_stim.Depth[s]))
corridor_df = pd.DataFrame(corridor_df)

fig = plt.figure()
fig.clf()
ax0 = fig.add_subplot(4, 1, 1)
ax0.plot(vis_stim.onix_time.values, vis_stim.Depth.values)
depths = corridor_df.depth.unique()
cmap = plt.get_cmap('Set2', len(depths))
depth_color = {d: cmap(i) for i, d in enumerate(depths)}

ax0.set_ylabel('Depth')
ax1 = fig.add_subplot(4, 1, 2, sharex=ax0)
ax1.plot(harp_message['analog_onix_time'], harp_message['mouse_speed'], alpha=0.2)
# make a moving average on a 100ms window (assuming 1kHz sampling)
w = int(0.5 * 1000)
cs = np.cumsum(harp_message['mouse_speed'])
smooth_speed = (cs[w:] - cs[:-w]) / w
ax1.plot(harp_message['analog_onix_time'][int(w/2):-int(w/2)], smooth_speed)
ax1.set_ylabel('Running speed')
ax1.set_ylim([-2, 3])
ax0.set_xlim([0, 3000])
for x in [ax0, ax1]:
    for i_c, cdf in corridor_df.iterrows():
        x.axvspan(cdf.start_time, cdf.end_time, alpha=0.2, color=depth_color[cdf.depth])


# make a firing rate map for all "good" units
bin_width = 1
bin_edges = np.arange(-120, corridor_df.end_time.max() + 120, bin_width)
# matrix will be in between the bins, so len(bin_edges) -
fr_matrix = np.zeros([len(good_units), len(bin_edges) - 2])
cluster_index_list = []
for index, (cluster_id, spikes) in enumerate(good_units.items()):
    cluster_index_list.append(cluster_id)
    fr_matrix[index] = np.diff(spikes.searchsorted(bin_edges))[1:]
# normalise each matrix in Z-score

zscore_mat = ((fr_matrix.T - fr_matrix.mean(axis=1)) / fr_matrix.std(axis=1)).T


ax2 = fig.add_subplot(2, 1, 2, sharex=ax0)
ax2.clear()
img = ax2.imshow(zscore_mat, aspect='auto', interpolation='None', origin='lower',
                 cmap='RdBu_r', vmin=-2, vmax=2,
                 extent=[bin_edges[1], bin_edges[-1], 0, zscore_mat.shape[0]])
cax = fig.add_subplot(2, 40, 80)
cb = fig.colorbar(img, cax=cax)
cb.set_label('z-score firing')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Unit #')