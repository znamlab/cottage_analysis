from pathlib import Path
import numpy as np
import pandas as pd
import flexiznam as flm
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.io_module.harp import load_harp

ONIX_DATA_FORMAT = dict(
    ephys="uint16", clock="uint64", aux="uint16", hubsynccounter="uint64", aio="uint16"
)


ONIX_SAMPLING = 250e6


RAW = Path(flm.PARAMETERS["data_root"]["raw"])
PROCESSED = Path(flm.PARAMETERS["data_root"]["processed"])

MAPPING = [
    39,
    37,
    35,
    33,
    47,
    45,
    43,
    41,
    55,
    53,
    51,
    49,
    57,
    63,
    61,
    59,
    62,
    60,
    58,
    56,
    54,
    52,
    50,
    48,
    46,
    44,
    42,
    40,
    38,
    36,
    34,
    32,
    24,
    26,
    28,
    30,
    16,
    18,
    20,
    22,
    8,
    10,
    12,
    14,
    0,
    2,
    4,
    6,
    3,
    5,
    7,
    1,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
]
# The mapping of electrode order as it comes out of the headstage to tetrodes 1 to 16.


def load_onix_recording(
    project,
    mouse,
    session,
    vis_stim_recording=None,
    harp_recording=None,
    onix_recording=None,
    allow_reload=True,
    raw_folder=RAW,
    processed_folder=PROCESSED,
    di_names=("lick_detection", "onix_clock", "di2_encoder_initial_state"),
    cut_if_not_multiple=False,
):
    """Main function calling all the subfunctions

    Args:
        project (str): name of the project
        mouse (str): name of the mouse
        session (str): name of the session
        vis_stim_recording (str): recording containing visual stimulation data
        harp_recording (str): recording containing harp data
        onix_recording (str): recording containing onix data
        allow_reload (bool): If True (default) will reload processed data instead of
                             raw when available
        di_names (list): list of DI names to load from HARP
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels if False, will load only if the data is a multiple
            of the number of channels. Default False.

    Returns:
        data (dict): a dictionary with one element per datasource
    """
    session_folder = raw_folder / project / mouse / session
    assert session_folder.is_dir()

    processed_folder = processed_folder / project / mouse / session
    out = dict()

    flm_sess = flm.get_flexilims_session(project)
    if harp_recording is not None:
        harp_message, harp_ds = synchronisation.load_harpmessage(
            recording="_".join([mouse, session, harp_recording]),
            flexilims_session=flm_sess,
            conflicts="skip" if allow_reload else "overwrite",
            di_names=di_names,
        )
        out["harp_message"] = dict(harp_message)

    if vis_stim_recording is not None:
        # add frame loggers and other CSVs
        out["vis_stim_log"] = load_vis_stim_log(session_folder / vis_stim_recording)

    if onix_recording is not None:
        # Load onix AI/DI
        breakout_data = load_breakout(session_folder / onix_recording)
        out["breakout_data"] = breakout_data
        try:
            out["rhd2164_data"] = load_rhd2164(
                session_folder / onix_recording, cut_if_not_multiple=cut_if_not_multiple
            )
        except IOError:
            print("Could not load RHD2164 data")
        try:
            out["ts4131_data"] = load_ts4231(session_folder / onix_recording)
        except IOError:
            print("Could not load TS4131 data")
    return out


def load_vis_stim_log(folder):
    out = dict()
    folder = Path(folder)
    for csv_file in folder.glob("*.csv"):
        what = csv_file.stem.split("_")[-1]
        out[what] = pd.read_csv(csv_file)

    return out


def load_rhd2164(
    path_to_folder,
    timestamp=None,
    num_chans=64,
    num_aux_chan=6,
    cut_if_not_multiple=False,
):
    """Load all files related to rhd2164, ie ephys

    Args:
        path_to_folder (str or Path): path to the folder containing ephys data
        timestamp (str or None): timestamp used in save name
        num_chans (int): number of ephys channels saved (default 64)
        num_aux_chan (int): number of auxiliary channels saved (default 6)
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels. if False, will load only if the data is a
            multiple of the number of channels. Default False.

    Returns:
        data dict: a dictionary of memmap
    """
    num_chan_dict = dict(ephys=num_chans, clock=1, aux=num_aux_chan, hubsynccounter=1)
    ephys_files = _find_files(path_to_folder, timestamp, "rhd2164")

    output = dict()
    for ephys_file in ephys_files:
        what = ephys_file.stem.split("_")[0][len("rhd2164-") :]
        if ephys_file.suffix == ".csv":
            assert what == "first-time"
            with open(ephys_file, "r") as f:
                output["first_time"] = f.read().strip()
            continue
        assert ephys_file.suffix == ".raw"

        data = _load_binary_file(
            ephys_file,
            dtype=ONIX_DATA_FORMAT[what],
            nchan=num_chan_dict[what],
            cut_if_not_multiple=cut_if_not_multiple,
        )
        output[what] = data
    return output


def reorder_array(ephys_data):
    """
    Reorder the rows of the ephys data based on a predefined mapping. This is useful because data does not come
    neatly ordered as [electrode 1 tetrode 1, electrode 2 tetrode 1, ..., electrode 4 tetrode 16] from the headstage.
    This function remaps inputs so that tetrodes are in order and remain together.

    Parameters:
    - ephys_data (np.ndarray): The ephys data to reorder. Usually, processed_ephys['ephys'].

    Returns:
    - np.ndarray: The reordered ephys data.
    """
    return ephys_data[MAPPING]


def load_ts4231(path_to_folder, timestamp=None):
    """Load data from the lighthouse system

    Args:
        path_to_folder (str or Path): path to the folder containing data
        timestamp (str or None): timestamp used in save name

    Returns:
        ts_out (dict): a dictionary of dataframe with one element per photodiode
    """

    ts_files = _find_files(path_to_folder, timestamp, "ts4231")
    ts_out = dict()
    for photodiode in ts_files:
        try:
            data = pd.read_csv(
                photodiode, header=0, names=["timestamp", "clock", "x", "y", "z"]
            )
        except pd.errors.EmptyDataError:
            continue
        ts_out[int(photodiode.stem.split("_")[0][len("ts4231-") :])] = data
    return ts_out


def load_breakout(
    path_to_folder, timestamp=None, num_ai_chan=2, cut_if_not_multiple=False
):
    """Load data from the breakout board, ie AI and DI

    Args:
        path_to_folder (str or Path): path to the folder containing breakout board data
        timestamp (str or None): timestamp used in save name
        num_ai_chan(int): number of analog input-output channels being recorded.
            In previous versions, it defaulted to 2. Now, the workflow saves how many it
            records and this function reads it. Keeping the default argument so that we
            are still able to read old sessions.
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels if False, will load only if the data is a multiple
            of the number of channels. Default False.

    Returns:
        data dict: a dictionary of memmap
    """
    breakout_files = _find_files(path_to_folder, timestamp, "breakout")
    output = dict()
    for breakout_file in breakout_files:
        what = breakout_file.stem.split("_")[0][len("breakout-") :]
        if breakout_file.suffix == ".csv":
            if what == "dio":
                dio = pd.read_csv(breakout_file)
                port = np.array(dio.Port.values, dtype="uint8")
                bits = np.unpackbits(port, bitorder="little")
                bits = bits.reshape((len(port), 8))
                for i in range(8):
                    dio["DI%d" % i] = bits[:, i]
                output["dio"] = dio
            if what == "analog":
                nchan = pd.read_csv(breakout_file)
                num_ai_chan = len(list(nchan))
        else:
            assert breakout_file.suffix == ".raw"
            if what == "aio-clock":
                nchan = 1
                dtype = ONIX_DATA_FORMAT["clock"]
            elif what == "aio":
                nchan = num_ai_chan
                dtype = ONIX_DATA_FORMAT["aio"]
            data = _load_binary_file(
                breakout_file,
                dtype=dtype,
                nchan=nchan,
                cut_if_not_multiple=cut_if_not_multiple,
            )
            output[what] = data
    return output


def load_bno055(
    path_to_folder,
    timestamp=None,
    num_chans_euler=3,
    num_chans_gravity=3,
    num_chans_linear_accel=3,
    num_chans_quaternion=4,
):
    """Loads the IMU data in a memmap dictionary
    Args:
        path_to_folder (str or Path): the full path to the folder which contains the IMU output.
        timestamp (str or None): timestamp used in save name

    Returns:
        bno_out: a dictionary of memmap
    """

    num_chan_dict = dict(
        euler=num_chans_euler,
        gravity=num_chans_gravity,
        linear=num_chans_linear_accel,
        quaternion=num_chans_quaternion,
    )

    bno_files = _find_files(path_to_folder, timestamp, "bno055")
    output = dict()
    for bno_file in bno_files:
        what = bno_file.stem.split("_")[0][len("bno055-") :]
        if "-" in what:
            what = Path(what)
            what = what.stem.split("-")[0]
        if bno_file.suffix == ".csv":
            other = pd.read_csv(bno_file)
            output["computer_timestamp"] = other.iloc[:, 0]
            output["onix_time"] = other.iloc[:, 1]
            output["temperature"] = other.iloc[:, 2]
            output["no_idea"] = other.iloc[:, 3]
            continue
        assert bno_file.suffix == ".raw"

        data = np.fromfile(bno_file, dtype=np.double).reshape(-1, num_chan_dict[what])

        output[what] = data
    return output


def load_camera_times(camera_dir):
    """
    Loads the metadata of the setup cameras.
    Args:
        camera_dir(str or Path): the complete path to the camera output directory
    Returns:
        output(dict): a dictionary containing one key per camera. Inside, a dictionary with the metadata of the camera.
    """
    camera_dir = Path(camera_dir)

    # Check if provided path is a directory
    if not camera_dir.is_dir():
        raise IOError(f"{camera_dir} is not a directory")

    # Search for all files containing the word 'camera' and ending with 'timestamps'
    camera_files = list(camera_dir.glob("*camera*timestamps*"))

    # If no valid files found, raise an error
    if not camera_files:
        raise IOError(f"Could not find any timestamp files in {camera_dir}")

    output = dict()
    seen_names = set()  # Set to track seen camera names

    for cam_file in camera_files:
        # Use the part before '_timestamps' as the key for the dictionary
        key = cam_file.stem.split("_timestamps")[0]

        # Check for duplicate names
        if key in seen_names:
            raise ValueError(f"Duplicate timestamp file detected for camera: {key}")

        seen_names.add(key)
        output[key] = pd.read_csv(cam_file)

    return output


def convert_ephys(
    uint16_file, target, nchan=64, overwrite=False, batch_size=1e6, verbose=True
):
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
        raise IOError("File %s already exists." % target)

    n_pts = uint16_file.stat().st_size / 2  # divide by 2 for uint16
    if np.mod(n_pts, nchan) != 0:
        raise IOError("Input data is not a multiple of %d" % nchan)
    n_time = int(n_pts / nchan)
    ephys_data = np.memmap(
        uint16_file, dtype="uint16", mode="r", order="F", shape=(nchan, n_time)
    )
    copy_data = np.memmap(
        target, dtype="int16", mode="w+", order="F", shape=(nchan, n_time)
    )

    ndone = 0
    if verbose:
        txt = "%.1f %%" % (ndone / n_time * 100)
        print(txt, flush=True)
    while ndone < n_time:
        end = min(ndone + batch_size, n_time)
        copy_data[:, ndone:end] = (
            np.array(ephys_data[:, ndone:end], dtype="int16") + 2**15
        )
        ndone = int(ndone + batch_size)
        if verbose:
            print("\b" * len(txt) + "%.1f %%" % (ndone / n_time * 100), flush=True)
            txt = "%.1f %%" % (ndone / n_time * 100)
    if verbose:
        print("Flushing to disk", flush=True)
    copy_data.flush()
    if verbose:
        print("done", flush=True)


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
        raise IOError("%s is not a directory" % folder)

    valid_files = list(folder.glob("%s*" % prefix))
    if not len(valid_files):
        raise IOError("Could not find any %s file in %s" % (prefix.upper(), folder))
    if timestamp is None:
        timestamp = "_".join(valid_files[0].stem.split("_")[1:])
        if not all([e.stem.endswith(timestamp) for e in valid_files]):
            raise IOError(
                "Multiple acquisition in folder %s. Specify timestamp" % folder
            )
    else:
        valid_files = [e for e in valid_files if e.stem.endswith(timestamp)]
    return valid_files


def _load_binary_file(file_path, dtype, nchan, order="F", cut_if_not_multiple=False):
    file_path = Path(file_path)
    n_pts = file_path.stat().st_size / np.dtype(dtype).itemsize
    if np.mod(n_pts, nchan) != 0:
        if cut_if_not_multiple:
            print(f"Warning: Data in {file_path} is not a multiple of {nchan}. Cutting")
            n_pts = int(n_pts // nchan * nchan)
        else:
            raise IOError("Data in %s is not a multiple of %d" % (file_path, nchan))
    n_time = int(n_pts / nchan)
    shape = (nchan, n_time) if nchan != 1 else None
    data = np.memmap(file_path, dtype=dtype, mode="r", order=order, shape=shape)
    return data
