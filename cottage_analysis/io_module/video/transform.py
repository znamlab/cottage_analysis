#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:08:43 2021

@author: hey2

Transform the shape of data
"""
import numpy as np
import os
import cottage_analysis.imaging.common.chunk_data as chunk_data


def transpose(
    data_folder,
    data_filename,
    data_shape,
    save_folder,
    save_filename,
    chunk_size,
    dtype=np.uint16,
    verbose=1,
):
    """
    Transpose a stored array (column major) into frame-major form.

    Parameters
    ----------
    data_folder : string
        Data folder
    data_filename : string
        Data filename
    data_shape : list, [height, width, frames]
        Data shape
    save_folder : string
        Save folder
    save_filename : string
        Save filename, can be in .bin or .npy
    chunk_size : int or None
        Bumber of frames for each chunk
    dtype : string, optional
        Data type. The default is 'float32'.
    verbose : 1 or 0, optional
        verbose or not. The default is 1.

    Returns
    -------
    None.

    """

    # load data into memmap
    data_path = data_folder + data_filename
    assert os.path.isfile(data_path)
    data = np.memmap(data_path, dtype=dtype, mode="r")

    # reshape data into (row, col, frame)
    height = data_shape[0]
    width = data_shape[1]
    frames = data_shape[2]
    data = data.reshape((height, width, -1), order="F")

    # create an empty memmap
    save_path = save_folder + save_filename
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    new_fp = np.memmap(save_path, dtype=dtype, mode="w+", shape=data.shape, order="C")

    # save frame-major ordered data to new memmap
    if chunk_size is not None:
        chunk_list = chunk_data.chunk_data(arr=np.arange(frames), chunk_size=chunk_size)

        for ichunk in range(0, len(chunk_list)):
            start_frame = chunk_list[ichunk][0]
            stop_frame = chunk_list[ichunk][-1] + 1
            new_fp[:, :, start_frame:stop_frame] = data[:, :, start_frame:stop_frame]
            new_fp.flush()

            if verbose == 1:
                print(
                    "finished: chunk " + str(ichunk) + "/" + str(len(chunk_list) - 1),
                    flush=True,
                )

    else:
        new_fp[:, :, :] = data[:, :, :]
        new_fp.flush()

    print("---Transpose finished.---", flush=True)
