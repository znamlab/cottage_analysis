import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import flexiznam as flm
from cottage_analysis.io_module import harp

ONIX_DATA_FORMAT = dict(ephys='uint16',
                        clock='uint64',
                        aux='uint16',
                        hubsynccounter='uint64',
                        aio='uint16')
BREAKOUT_DIGITAL_INPUTS = dict(DI0='fm_cam_trig',
                               DI1='oni_clock_di',
                               DI2='hf_cam_trig')
ONIX_SAMPLING = 250e6
ENCODER_CPR = 4096
WHEEL_DIAMETER = 20e-2  # wheel diameter in meters

RAW = Path(flm.PARAMETERS['data_root']['raw'])
PROCESSED = Path(flm.PARAMETERS['data_root']['processed'])


def load_onix_recording(project, mouse, session, vis_stim_recording=None,
                        onix_recording=None, allow_reload=True,
                        breakout_di_names=BREAKOUT_DIGITAL_INPUTS):
    """Main function calling all the subfunctions

    Args:
        project (str): name of the project
        mouse (str): name of the mouse
        session (str): name of the session
        vis_stim_recording (str): recording containing visual stimulation data
        onix_recording (str): recording containing onix data
        allow_reload (bool): If True (default) will reload processed data instead of
                             raw when available
        breakout_di_names (dict): Names of DI on breakout board, e.g. {'DI0': 'lick'}

    Returns:
        data (dict): a dictionary with one element per datasource
    """
    session_folder = RAW / project / mouse / session
    assert session_folder.is_dir()

    processed_folder = PROCESSED / project / mouse / session
    out = dict()

    if vis_stim_recording is not None:
        harp_message = '%s_%s_%s_harpmessage.bin' % (mouse, session, vis_stim_recording)
        raw_harp = session_folder / vis_stim_recording / harp_message
        processed_messages = processed_folder / vis_stim_recording / (raw_harp.stem + '.npz')
        processed_messages.parent.mkdir(exist_ok=True, parents=True)
        if allow_reload and processed_messages.is_file():
            harp_message = dict(np.load(processed_messages))
        else:
            # slow part: read harp messages so save output and reload
            harp_message = load_harp(raw_harp)
            np.savez(processed_messages, **harp_message)
        out['harp_message'] = harp_message
        # add frame loggers and other CSVs
        out['vis_stim_log'] = load_vis_stim_log(session_folder / vis_stim_recording)

    if onix_recording is not None:
        # Load onix AI/DI
        breakout_data = load_breakout(session_folder / onix_recording)
        # use human readable names
        breakout_data['dio'].rename(columns=breakout_di_names, inplace=True)
        out['breakout_data'] = breakout_data
        try:
            out['rhd2164_data'] = load_rhd2164(session_folder / onix_recording)
        except IOError:
            print('Could not load RHD2164 data')
        try:
            out['ts4131_data'] = load_ts4231(session_folder / onix_recording)
        except IOError:
            print('Could not load TS4131 data')
    return out


def load_vis_stim_log(folder):
    out = dict()
    folder = Path(folder)
    for csv_file in folder.glob('*.csv'):
        what = csv_file.stem.split('_')[-1]
        out[what] = pd.read_csv(csv_file)

    return out

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
    di = harp_message[harp_message.address == 32]
    if len(di):
        bits = np.array(np.hstack(di.data.values), dtype='uint8')
        bits = np.unpackbits(bits, bitorder='little')
        bits = bits.reshape((len(di), 8))

        # keep only digital input
        names = ['lick_detection', 'onix_clock', 'di2_encoder_initial_state']
        bits = {names[n]: bits[:, n] for n in range(3)}
        output.update(bits)
        output['digital_time'] = di.timestamp_s.values
    else:
        warnings.warn('Could not find any digital input!')

    # make a speed out of rotary increment
    mvt = np.diff(output['rotary'])
    rollover = np.abs(mvt) > 40000
    mvt[rollover] -= 2 ** 16 * np.sign(mvt[rollover])
    # The rotary count decreases when the mouse goes forward
    mvt *= -1
    # 0-padding to keep constant length
    dst = np.array(np.hstack([0, mvt]), dtype=float)
    wheel_gain = WHEEL_DIAMETER / 2 * np.pi * 2 / ENCODER_CPR
    output['rotary_meter'] = dst * wheel_gain
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
    num_chan_dict = dict(ephys=num_chans, clock=1, aux=num_aux_chan, hubsynccounter=1)
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

        data = _load_binary_file(ephys_file,
                                 dtype=ONIX_DATA_FORMAT[what],
                                 nchan=num_chan_dict[what])
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


def load_breakout(path_to_folder, timestamp=None, num_ai_chan=2):
    """Load data from the breakout board, ie AI and DI

    Args:
        path_to_folder (str or Path): path to the folder containing breakout board data
        timestamp (str or None): timestamp used in save name
        num_ai_chans (int): number of ephys channels saved (default 64)

    Returns:
        data dict: a dictionary of memmap
    """
    breakout_files = _find_files(path_to_folder, timestamp, 'breakout')
    output = dict()
    for breakout_file in breakout_files:
        what = breakout_file.stem.split('_')[0][len('breakout-'):]
        if breakout_file.suffix == '.csv':
            assert what == 'dio'
            dio = pd.read_csv(breakout_file, )
            port = np.array(dio.Port.values, dtype='uint8')
            bits = np.unpackbits(port, bitorder='little')
            bits = bits.reshape((len(port), 8))
            for i in range(8):
                dio['DI%d' % i] = bits[:, i]
            output['dio'] = dio
            continue
        assert breakout_file.suffix == '.raw'
        if what == 'aio-clock':
            nchan = 1
            dtype = ONIX_DATA_FORMAT['clock']
        elif what == 'aio':
            nchan = num_ai_chan
            dtype = ONIX_DATA_FORMAT['aio']
        data = _load_binary_file(breakout_file, dtype=dtype, nchan=nchan)
        output[what] = data
    return output


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
    return valid_files


def _load_binary_file(file_path, dtype, nchan):
    file_path = Path(file_path)
    n_pts = file_path.stat().st_size / np.dtype(dtype).itemsize
    if np.mod(n_pts, nchan) != 0:
        raise IOError('Data in %s is not a multiple of %d' % (file_path, nchan))
    n_time = int(n_pts / nchan)
    shape = (nchan, n_time) if nchan != 1 else None
    data = np.memmap(file_path, dtype=dtype, mode='r', order='F',
                     shape=shape)
    return data
