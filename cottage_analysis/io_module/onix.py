from pathlib import Path

import numpy as np


def convert_ephys(uint16_file, target, nchan=64, overwrite=False, batch_size=1e6,
                  verbose=True):
    """Convert raw uint16 data in int16

    Data from onix is saved as uint16. Kilosort has no option to change expected
    datatype and expects int16. This function copies the data to the new file changing the
    datatype.

    Args:
        uint16_file (str or Path): path to the raw data (F order, uint16)
        target (str or Path): target to write the new data
        nchan (int): number of channels (default False)
        overwrite (bool): overwrite target if it exists (default False)
        batch_size (int): number of time points to process at once
        verbose (bool): print progress (default True)

    Returns:
        None
    """
    uint16_file = Path(uint16_file)
    target = Path(target)
    if target.is_file() and (not overwrite):
        raise IOError('File %s already exists.' % target)

    n_pts = uint16_file.stat().st_size / 2  # divide by 2 for uint16
    if np.mod(n_pts, nchan) != 0:
        raise IOError('Input data is not a multiple of %d' % nchan)
    n_time = int(n_pts / nchan)
    ephys_data = np.memmap(uint16_file, dtype='uint16', mode='r', order='F',
                           shape=(nchan, n_time))
    copy_data = np.memmap(target, dtype='int16', mode='w+', order='F',
                          shape=(nchan, n_time))

    ndone = 0
    if verbose:
        txt = '%.1f %%' % (ndone / n_time * 100)
        print(txt, flush=True)
    while ndone < n_time:
        end = min(ndone + batch_size, n_time)
        copy_data[:, ndone:end] = np.array(ephys_data[:, ndone:end],
                                           dtype='int16') + 2 ** 15
        ndone = ndone + batch_size
        if verbose:
            print('\b' * len(txt) + '%.1f %%' % (ndone / n_time * 100), flush=True)
            txt = '%.1f %%' % (ndone / n_time * 100)
    if verbose:
        print('Flushing to disk', flush=True)
    copy_data.flush()
    if verbose:
        print('done', flush=True)
