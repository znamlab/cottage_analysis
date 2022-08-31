from pathlib import Path
import numpy as np
import pandas as pd
from cottage_analysis.io_module import harp

RHD2164_DATA_FORMAT = dict(ephys='uint16',
                           clock='uint64',
                           aux='uint16')


def load_harp(harp_bin):
    # Harp
    harp_message = harp.read_message(path_to_file=harp_bin)
    harp_message = pd.DataFrame(harp_message)
    output = dict()

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
    output['reward_times'] = reward_message.timestamp_s.values

    # EVENT messages are analog and digital input.
    # Analog are the photodiode and the rotary encoder, both on address 44
    analog = harp_message[harp_message.address == 44]
    harp_analog_times = analog.timestamp_s.values
    analog = np.vstack(analog.data)

    output['analog_time'] = harp_analog_times
    output['rotary'] = analog[:, 1]
    output['photodiode'] = analog[:, 0]

    # Digital input is on address 32, the data is 2 when the trigger is high
    di = harp_message[(harp_message.address == 32) & (harp_message.data == (2,))]
    output['onix_clock'] = di.timestamp_s.values
    return output


def load_rhd2164(path_to_folder, timestamp=None, num_chans=64, num_aux_chan=6):
    """Load all files related to rhd2164, ie ephys

    Args:
        path_to_folder (str or Path): path to the folder containing ephys data
        timestamp (str or None): timestamp used in save name
        num_chans (int): number of ephys channels saved (default 64)
        num_aux_chan (int): number of auxiliary channels saved (default 6)

    Returns:
        data dict: a dictionary of memmap
    """
    num_chan_dict = dict(ephys=num_chans, clock=1, aux=num_aux_chan)
    ephys_files = _find_files(path_to_folder, timestamp, 'rhd2164')

    output = dict()
    for ephys_file in ephys_files:
        what = ephys_file.stem.split('_')[0][len('rhd2164-'):]
        if ephys_file.suffix == '.csv':
            assert what == 'first-time'
            with open(ephys_file, 'r') as f:
                output['first_time'] = f.read().strip()
            continue
        assert ephys_file.suffix == '.raw'
        dtype = np.dtype(RHD2164_DATA_FORMAT[what])
        n_pts = ephys_file.stat().st_size / dtype.itemsize
        nchan = num_chan_dict[what]
        if np.mod(n_pts, nchan) != 0:
            raise IOError('%s data is not a multiple of %d' % (what, nchan))
        n_time = int(n_pts / nchan)
        data = np.memmap(ephys_file, dtype=dtype, mode='r', order='F',
                         shape=(nchan, n_time))
        output[what] = data
    return output


def load_ts4231(path_to_folder, timestamp=None):
    """Load data from the lighthouse system

    Args:
        path_to_folder (str or Path): path to the folder containing data
        timestamp (str or None): timestamp used in save name

    Returns:
        ts_out (dict): a dictionary of dataframe with one element per photodiode
    """

    ts_files = _find_files(path_to_folder, timestamp, 'ts4231')
    ts_out = dict()
    for photodiode in ts_files:
        try:
            data = pd.read_csv(photodiode, header=0,
                               names=['timestamp', 'clock', 'x', 'y', 'z'])
        except pd.errors.EmptyDataError:
            continue
        ts_out[int(photodiode.stem.split('_')[0][len('ts4231-'):])] = data
    return ts_out


def convert_ephys(uint16_file, target, nchan=64, overwrite=False, batch_size=1e6,
                  verbose=True):
    """Convert raw uint16 data in int16

    Data from onix is saved as uint16. Kilosort has no option to change expected
    datatype and expects int16. This function copies the data to the new file changing the
    datatype.

    Args:
        uint16_file (str or Path): path to the raw data (F order, uint16)
        target (str or Path): target to write the new data
        nchan (int): number of channels (default 64)
        overwrite (bool): overwrite target if it exists (default False)
        batch_size (int): number of time points to process at once (default 1e6)
        verbose (bool): print progress (default True)

    Returns:
        None
    """
    uint16_file = Path(uint16_file)
    batch_size = int(batch_size)  # force int to be able to use for indexing
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
        ndone = int(ndone + batch_size)
        if verbose:
            print('\b' * len(txt) + '%.1f %%' % (ndone / n_time * 100), flush=True)
            txt = '%.1f %%' % (ndone / n_time * 100)
    if verbose:
        print('Flushing to disk', flush=True)
    copy_data.flush()
    if verbose:
        print('done', flush=True)


def _find_files(folder, timestamp, prefix):
    """Inner function to return list of files with filter_name and timestamp

    Args:
        folder(str or Path): path to the folder containing data
        timestamp (str or None): timestamp used in save name
        prefix (str): prefix filter

    Returns:
        file_list (list): list of valid files
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise IOError('%s is not a directory' % folder)

    valid_files = list(folder.glob('%s*' % prefix))
    if not len(valid_files):
        raise IOError('Could not find any %s file in %s' % (prefix.upper(), folder))
    if timestamp is None:
        timestamp = '_'.join(valid_files[0].stem.split('_')[1:])
        if not all([e.stem.endswith(timestamp) for e in valid_files]):
            raise IOError('Multiple acquisition in folder %s. Specify timestamp' % folder)
    else:
        valid_files = [e for e in valid_files if e.stem.endswith(timestamp)]
    return  valid_files
