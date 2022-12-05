"""
Test of preprocessing of sphere protocol
"""

import time
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from cottage_analysis.preprocessing import find_frames as ff
from cottage_analysis.io_module import harp
import pandas as pd
import flexiznam as flm
from scipy.sparse import bsr_array
from cottage_analysis.stimulus_structure import spheres_tube as stu
from cottage_analysis.utilities.time_series_analysis import searchclosest
from cottage_analysis.utilities import continuous_data_analysis as cda

PROJECT = 'blota_onix_pilote'
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220831'
RECORDING = 'R163332_SpheresPermTubeReward'
MESSAGES = '%s_%s_%s_harpmessage.bin' % (MOUSE, SESSION, RECORDING)
FRAME_LOG = '%s_%s_%s_FrameLog.csv' % (MOUSE, SESSION, RECORDING)
PARAMS_LOG = '%s_%s_%s_NewParams.csv' % (MOUSE, SESSION, RECORDING)
ROTARY_LOG = '%s_%s_%s_RotaryEncoder.csv' % (MOUSE, SESSION, RECORDING)


def test_recreate_stimulus():
    # load data
    data_root = flm.PARAMETERS['data_root']
    raw_data = Path(data_root['raw']) / PROJECT / MOUSE / SESSION / RECORDING
    msg = raw_data / MESSAGES
    p_msg = Path(data_root['processed']) / PROJECT / MOUSE / SESSION / RECORDING
    p_msg = p_msg / (msg.stem + '.npz')
    if p_msg.is_file():
        harp_messages = np.load(p_msg)
    else:
        harp_messages = harp.load_harp(msg)
        p_msg.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p_msg, **harp_messages)
    sphere_size = 10
    azimuth_limits = [-120, 120]
    elevation_limits = [-40, 40]
    resolution = 1
    time_column = 'HarpTime'

    frame_log = pd.read_csv(raw_data / FRAME_LOG)
    params = pd.read_csv(raw_data / PARAMS_LOG)
    rot_log = pd.read_csv(raw_data / ROTARY_LOG)
    corridor_df = stu.trial_structure(params, time_column=time_column)
    all_frames = frame_log['HarpTime']

    mouse_pos_cm = harp_messages['rotary_meter'].cumsum() * 100
    frame_times = all_frames[::10][20000:22000]
    out_shape = (len(frame_times), int(np.diff(elevation_limits) / resolution),
                 int(np.diff(azimuth_limits) / resolution))
    outsize = np.prod(out_shape)*2/1024**3
    if outsize > 10:
        print('big image: %.2f Gb' % outsize)
    output = np.zeros(out_shape, dtype='int16')
    start_time = time.time()
    frames = stu.regenerate_frames(frame_times,
                                   params,
                                   mouse_pos_cm,
                                   mouse_pos_time=harp_messages['analog_time'],
                                   corridor_df=corridor_df, time_column='HarpTime',
                                   resolution=resolution, sphere_size=sphere_size,
                                   azimuth_limits=(-120, 120),
                                   elevation_limits=(-40, 40),
                                   output=output)
    end_time = time.time()
    print('That took %s s' % (end_time-start_time))
    fig = plt.figure()
    plt.imshow(frames.mean(axis=0), extent=azimuth_limits+elevation_limits)
    fig.savefig(str(p_msg.with_name('temp.png')))


if __name__ == '__main__':
    test_recreate_stimulus()