import os
import struct
import cv2
import numpy as np
import mmap
import pandas as pd

DEPTH_DICT = {8: np.uint8,
              16: np.uint16}


def load_video(data_folder, camera, ordered=True):
    """Load the video from an eye cam"""
    metadata_file = os.path.join(data_folder, '%s_metadata.txt' % camera)
    assert os.path.isfile(metadata_file)
    metadata = {}
    with open(metadata_file, 'r') as m_raw:
        for line in m_raw:
            if line.strip():
                k, v = line.strip().split(":")
                metadata[k.strip()] = int(v.strip())

    binary_file = os.path.join(data_folder, '%s_data.bin' % camera)
    assert os.path.isfile(binary_file)
    data = np.memmap(binary_file, dtype=DEPTH_DICT[metadata['Depth']], mode='r')
    
    if ordered == True:        
        data = data.reshape((metadata['Height'], metadata['Width'], -1), order='F')
    return data


def write_array_to_video(target_file, video_array, frame_rate, is_color=False, verbose=True,
                         codec='mp4v', extension='.mp4', min_brightness=None, max_brightness=None,
                         perc_saturation=0, overwrite=False):
    """Write an array to a mp4 file

    The array must shape must be (lines/height x columns/width x frames)
    """
    if not target_file.endswith(extension):
        target_file += extension
    if not overwrite:
        assert not os.path.isfile(target_file)
    if verbose:
        print('Creating writer')
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # writer to `output` in mp4v at 20 fps of the correct size, NOT color
    out = cv2.VideoWriter(target_file, fourcc, frame_rate, (video_array.shape[1], video_array.shape[0]), is_color)

    # rescale contrast
    if verbose:
        print('Finding saturation limits')
    if perc_saturation:
        sample_hist = np.percentile(video_array, [perc_saturation/2, 100-perc_saturation/2])
    else:
        sample_hist = [0, 255]
        if min_brightness is not None:
            sample_hist[0] = min_brightness
        if max_brightness is not None:
            sample_hist[1] = max_brightness
    if verbose:
        msg = ''
        n_frames = video_array.shape[2]
        frames_done = 0
    for frame in range(video_array.shape[2]):
        frame = np.array(video_array[:, :, frame])
        if min_brightness is not None:
            frame[frame < min_brightness] = min_brightness
        if max_brightness is not None:
            frame[frame > max_brightness] = max_brightness
        if sample_hist[0] != 0 or sample_hist[1] != 255:
            rescaled_frame = (np.array(frame, dtype=float) - sample_hist[0]) / sample_hist[1] * 255
            rescaled_frame[rescaled_frame > 255] = 255
            rescaled_frame[rescaled_frame < 0] = 0
            rescaled_frame = np.array(rescaled_frame, dtype=np.uint8)
        else:
            rescaled_frame = frame
        out.write(rescaled_frame)
        frames_done += 1
        if verbose:
            erase_line = '\b' * len(msg)
            msg = '%d/%d frames done (%d%%)' % (frames_done, n_frames, frames_done/n_frames*100)
            print(erase_line + msg, flush=True, end='')
    if verbose:
        print('Done!')
    out.release()
    return