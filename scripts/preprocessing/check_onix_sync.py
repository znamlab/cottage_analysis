"""
A script to give detailed output of sync on the onix setup

This test:
- if camera trigger can be register
- if frames from head-fixed can be detected
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import flexiznam as flm
from scipy import signal

import cottage_analysis.io_module.harp
from cottage_analysis.ephys import continuous_analysis as ca
from cottage_analysis.ephys.synchronisation import plot_onix_harp_clock_sync
from cottage_analysis.imaging.common import find_frames
from cottage_analysis.io_module import onix

PROJECT = 'blota_onix_pilote'
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220831'
RECORDING = 'R163359'
VIS_STIM = 'R163332_SpheresPermTubeReward'

data_root = flm.PARAMETERS['data_root']
raw = Path(data_root['raw']) / PROJECT / MOUSE / SESSION
processed = Path(data_root['processed']) / PROJECT / MOUSE / SESSION

save_root = processed / RECORDING / 'sync_check'
save_root.mkdir(exist_ok=True)

#####################
##### LOAD DATA #####
#####################

ONIX_SAMPLING = 250e6
ENCODER_CPR = 4096
WHEEL_DIAMETER = 20

# Load harp messages
processed_messages = processed / VIS_STIM / 'processed_harp.npz'
processed_messages.parent.mkdir(exist_ok=True)
if Path(processed_messages).is_file():
    harp_message = dict(np.load(processed_messages))
else:
    # slow part: read harp messages so save output and reload
    harp_message = cottage_analysis.io_module.harp.load_harp(raw / VIS_STIM / ('%s_%s_%s_harpmessage.bin'
                                                                               % (MOUSE, SESSION, VIS_STIM)))
    np.savez(processed_messages, **harp_message)
# make a speed out of rotary increament
mvt = np.diff(harp_message['rotary'])
rollover = np.abs(mvt) > 40000
mvt[rollover] -= 2**16 * np.sign(mvt[rollover])
# The rotary count decreases when the mouse goes forward
mvt *= -1
harp_message['mouse_speed'] = np.hstack([0, mvt])  # 0-padding to keep constant length
wheel_gain = WHEEL_DIAMETER/2 * np.pi * 2 / ENCODER_CPR

# Load onix AI/DI
breakout_data = onix.load_breakout(raw / RECORDING)

dio = breakout_data['dio']
fm_cam_trig = dio.DI0
oni_clock_di = dio.DI1
hf_cam_trig = dio.DI0

#####################
#### SYNCHRONISE ####
#####################

# HARP to ONIX time
real_time = np.arange(len(harp_message['onix_clock'])) * 10e-3  # assume a perfect 100Hz
# linear regression:
t0 = harp_message['onix_clock'][0]
slope = np.nanmean(real_time[1:] / (harp_message['onix_clock'][1:] - t0))
def harp2onix(data):
    """Convert harp timestamp in onix time"""
    return (data - t0) * slope
harp_message['analog_time_onix'] = harp2onix(harp_message['analog_time'])

# Frame number to ONIX time
# Photodiode frame detection
harp_ai_sampling = 1/np.nanmedian(np.diff(harp_message['analog_time_onix']))
zsc_hf_pd = harp_message['photodiode']
zsc_hf_pd = (zsc_hf_pd - np.nanmean(zsc_hf_pd)) / np.nanstd(zsc_hf_pd)
df = np.diff(np.sign(zsc_hf_pd)) != 0
frame_start = harp_message['analog_time_onix'][1:][df]


s = harp_message['mouse_speed']
t = harp_message['analog_time_onix']
sampling = 1/np.nanmedian(np.diff(t))
cs = s.cumsum()
w = 20
binned = (cs[w:] - cs[:-w])[::w]
cm = np.array(binned, dtype=float) * wheel_gain
rate = cm / (w / sampling)
plt.subplot(2, 1, 1)
plt.plot(t[int(w/2):-int(w/2):w], rate)
plt.title('Speed in %d ms bin' % (w / sampling * 1000))
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.subplot(2, 2, 3)
plt.hist(rate, bins=np.arange(-5, 30))
plt.xlabel('Speed (cm/s)')
plt.subplot(2, 2, 4)
plt.hist(rate, bins=np.arange(-5, 30))
plt.yscale('log')
plt.xlabel('Speed (cm/s)')
plt.show()

photodiode_df = pd.DataFrame(dict(Photodiode=zsc_hf_pd,
                                  HarpTime=harp_message['analog_time_onix']))
frame_starts = find_frames.find_VS_frames(photodiode_df, frame_rate=144, upper_thr=0.5,
                                          lower_thr=-0.5, plot=True, plot_start=1000,
                                          plot_range=200, plot_dir=save_root)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
w = int(0.3 * harp_ai_sampling)
s = int(60 * harp_ai_sampling)
ax.plot(harp_message['analog_time_onix'][s:s+w], zsc_hf_pd[s:s+w])
ax.plot(frame_starts['HarpTime'], frame_starts['Photodiode'], 'o')
ax.set_xlim(*harp_message['analog_time_onix'][[s, s + w]])
ax.set_xlabel('ONI time (s)')
ax.set_ylabel('Photodiode (z-score)')

ax = fig.add_subplot(2, 2, 2)
ax.hist(np.diff(frame_starts['HarpTime']) * 1000)
ax.set_xlabel(r'$\Delta$ frame (ms)')
ax.set_yscale('log')

ax = fig.add_subplot(2, 2, 3)
ax.hist(np.diff(frame_starts['HarpTime']) * 1000, bins=np.arange(40))
ax.set_xlabel(r'$\Delta$ frame (ms)')
ax.set_yscale('log')

dt = np.diff(frame_starts['HarpTime']) * 1000
st = frame_starts.iloc[1:][dt > 10].iloc[0] - 0.05
w = int(0.1 * harp_ai_sampling)
s = int(st.HarpTime * harp_ai_sampling)

ax = fig.add_subplot(2, 2, 4)
ax.plot(harp_message['analog_time_onix'][s:s+w], zsc_hf_pd[s:s+w])
ax.plot(frame_starts['HarpTime'], frame_starts['Photodiode'], 'o')
ax.set_xlim(*harp_message['analog_time_onix'][[s, s + w]])
ax.set_xlabel('ONI time (s)')
ax.set_ylabel('Photodiode (z-score)')
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.savefig(save_root / 'frame_detecting.png', dpi=600)
plt.show()


######################
### DO SOME CHECKS ###
######################
fig = plot_onix_harp_clock_sync(oni_clock_di=oni_clock_di,
                                oni_clock_times=dio.Clock.values,
                                oni_clock_in_harp=harp_message['onix_clock'],
                                harp2onix=harp2onix,
                                onix_sampling=ONIX_SAMPLING)
fig.savefig(save_root / 'onix_clock_harp_sync.png', dpi=600)
fig.show()

###############################
# cut only the HF part
aio_time = breakout_data['aio-clock']
print('Finding limits', flush=True)
harp_time = harp_message['analog_onix_time']
resampled = aio_time.searchsorted(harp_time * onix_sampling)
print('Cutting time', flush=True)
aio_time = np.array(aio_time[resampled], dtype=float) / onix_sampling
print('Cutting data', flush=True)
aio_rs = np.array(breakout_data['aio'][1, resampled], dtype=float)
aio_rs -= np.quantile(aio_rs, 0.05)
aio_rs /= aio_rs.max()

print('Done', flush=True)

harp_rs = np.array(harp_message['mouse_speed'], dtype=float)
c = harp_rs.cumsum()
w = 10
smooth = (c[w:] - c[:-w]) / w
smooth *= wheel_gain

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(harp_time * 1000, harp_rs, label='encoder ticks')
ax.plot(harp_time[w:] * 1000, smooth, label='smoothed over %d ms' % w)
ax.plot(aio_time * 1000, aio_rs, label='Photodiode signal', color='purple')
ax.set_xlim([0, 1000])
ax.set_ylim([-0.1, 1.1])
ax.legend(loc=0)
ax.set_label('Time (ms)')
ax = fig.add_subplot(2, 1, 2)
corr = signal.correlate(aio_rs[w:], smooth)
lags = signal.correlation_lags(len(aio_rs[w:]), len(smooth))
dt = np.median(np.diff(harp_time))
m = corr.argmax()
tmax = lags[m] * dt
print(tmax * 1000)
ax.axvline(tmax * 1000)
ax.plot(lags*dt * 1000, corr)
#ax.set_xlim([-100, 100])
#ax.set_ylim([1e5, 2e5])
ax.set_xlabel(r'$\Delta$ time')
ax.set_ylabel('Correlation')
plt.show()