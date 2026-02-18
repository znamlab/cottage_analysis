import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation

ONIX_DATA_FORMAT = dict(
    ephys="uint16", clock="uint64", aux="uint16", hubsynccounter="uint64", aio="float32"
)
ONIX_SAMPLING = 250e6


def load_onix(
    onix_ds,
    project=None,
    flexilims_session=None,
    cut_if_not_multiple=False,
    ignore_wrong_timestamps=False,
):
    """Main function calling all the subfunctions

    Args:
        onix_ds (flexiznam.schema.onix_data.OnixData or str): Onix dataset or its name
        project (str, optional): Name of the project. Defaults to None.
        flexilims_session (flexilims.session, optional): Flexilims session. Defaults to
            None. Must be provided if project is None and onix_ds has no.
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels if False, will load only if the data is a multiple
            of the number of channels. Default False.

    Returns:
        data (dict): a dictionary with one element per datasource
    """
    if isinstance(onix_ds, str):
        if flexilims_session is None:
            flexilims_session = flz.get_flexilims_session(project)
        onix_ds = flz.Dataset.from_flexilims(
            name=onix_ds, flexilims_session=flexilims_session
        )

    out = dict()
    # Load onix AI/DI
    breakout_data = load_breakout(
        onix_ds.path_full,
        cut_if_not_multiple=cut_if_not_multiple,
        ignore_wrong_timestamps=ignore_wrong_timestamps,
    )
    out["breakout_data"] = breakout_data
    try:
        out["rhd2164_data"] = load_rhd2164(
            onix_ds.path_full,
            cut_if_not_multiple=cut_if_not_multiple,
            ignore_wrong_timestamps=ignore_wrong_timestamps,
        )
    except IOError:
        print("Could not load RHD2164 data")
    try:
        out["ts4131_data"] = load_ts4231(
            onix_ds.path_full, ignore_wrong_timestamps=ignore_wrong_timestamps
        )
    except IOError:
        print("Could not load TS4131 data")
    try:
        out["bno055_data"] = load_bno055(
            onix_ds.path_full,
            cut_if_not_multiple=cut_if_not_multiple,
            ignore_wrong_timestamps=ignore_wrong_timestamps,
        )
    except IOError:
        print("Could not load BNO055 data")

    return out


def load_rhd2164(
    path_to_folder,
    timestamp=None,
    num_chans=64,
    num_aux_chan=6,
    cut_if_not_multiple=False,
    ignore_wrong_timestamps=False,
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
        ignore_wrong_timestamps (bool): if True and timestamp is None, will keep all
            files with the prefix without looking at timestamps, if False, will raise an
            error if multiple timestamps are found. Default False.

    Returns:
        data dict: a dictionary of memmap
    """
    num_chan_dict = dict(ephys=num_chans, clock=1, aux=num_aux_chan, hubsynccounter=1)
    ephys_files = _find_files(
        path_to_folder,
        timestamp,
        "rhd2164",
        ignore_wrong_timestamps=ignore_wrong_timestamps,
    )

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


def load_neuropixel(path_to_folder, index=None, cut_if_not_multiple=True):
    """
    Load neuropixel data from the onix folder

    Args:
        path_to_folder (str or Path): path to the folder containing neuropixel data
        index (int or None): index of the neuropixel probe
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels. if False, will load only if the data is a
            multiple of the number of channels. Default True.

    Returns:
        data dict: a dictionary of memmap
    """
    path_to_folder = Path(path_to_folder)
    if index is not None:
        index = int(index)
    else:
        ephys_files = list(path_to_folder.glob("np2-*.raw"))
        indices = [int(f.stem.split("_")[1]) for f in ephys_files]
        indices = list(set(indices))
        if len(indices) > 1:
            raise OSError("Multiple neuropixel index files found")
        index = indices[0]
    output = dict()
    for probe in ["np2-a", "np2-b"]:
        ephys_file = path_to_folder / f"{probe}-ephys_{index}.raw"
        if not ephys_file.exists():
            continue
        data = _load_binary_file(
            ephys_file,
            dtype="uint16",
            nchan=384,
            cut_if_not_multiple=cut_if_not_multiple,
        )
        clock = _load_binary_file(
            path_to_folder / f"{probe}-clock_{index}.raw",
            dtype=ONIX_DATA_FORMAT["clock"],
            nchan=1,
            cut_if_not_multiple=cut_if_not_multiple,
        )
        output[probe] = dict(ephys=data, clock=clock)
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


def load_ts4231(path_to_folder, timestamp=None, ignore_wrong_timestamps=False):
    """Load data from the lighthouse system

    Args:
        path_to_folder (str or Path): path to the folder containing data
        timestamp (str or None): timestamp used in save name
        ignore_wrong_timestamps (bool): if True and timestamp is None, will keep all
            files with the prefix without looking at timestamps, if False, will raise an
            error if multiple timestamps are found. Default False.

    Returns:
        ts_out (dict): a dictionary of dataframe with one element per photodiode
    """

    ts_files = _find_files(
        path_to_folder,
        timestamp,
        "ts4231",
        ignore_wrong_timestamps=ignore_wrong_timestamps,
    )
    ts_out = dict()
    for photodiode in ts_files:
        try:
            data = pd.read_csv(
                photodiode,
                header=0,
                names=["clock", "x", "y", "z", "timestamp"],
                parse_dates=["timestamp"],
            )
        except pd.errors.EmptyDataError:
            continue
        ts_out[int(photodiode.stem.split("_")[0][len("ts4231-") :])] = data
    return ts_out


def load_breakout(
    path_to_folder,
    index=None,
    cut_if_not_multiple=False,
    verbose=True,
):
    """Load data from the breakout board, ie AI and DI

    Args:
        path_to_folder (str or Path): path to the folder containing breakout board data
        index (int, None): index of the acquistion
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels if False, will load only if the data is a multiple
            of the number of channels. Default False.
        verbose (bool): if True, print information about the data being loaded. Default True.

    Returns:
        data dict: a dictionary of memmap
    """
    path_to_folder = Path(path_to_folder)

    if index is None:
        # Match all analog-data files to find indices
        indices = []
        for match in path_to_folder.glob("analog-data_*.raw"):
            indices.append(match.stem.split("_")[1])
        if len(indices) == 0:
            raise IOError("No analog-data files found in %s" % path_to_folder)
        elif len(indices) > 1:
            raise IOError("Multiple analog-data files found in %s" % path_to_folder)
        index = int(indices[0])

    meta = pd.read_csv(
        path_to_folder / f"start-time_{index}.csv",
        names=["start_time", "acq_clk_hz", "block_read_sz", "block_write_sz"],
        skiprows=1,
        converters=dict(start_time=pd.to_datetime),
        dtype=dict(
            acq_clk_hz=np.uint32, block_read_sz=np.uint32, block_write_sz=np.uint32
        ),
    ).iloc[0]

    if verbose:
        print(f"Recording was started at {meta['start_time']} GMT")

    output = dict(meta)
    # first I need to find aio-channels to count the number of channels
    analog_channels = np.loadtxt(
        path_to_folder / f"analog-channels_{index}.csv", dtype="int8", delimiter=","
    )
    output["aio-channels"] = analog_channels
    num_ai_chan = len(analog_channels)
    output["aio"] = _load_binary_file(
        path_to_folder / f"analog-data_{index}.raw",
        dtype=ONIX_DATA_FORMAT["aio"],
        nchan=num_ai_chan,
        cut_if_not_multiple=cut_if_not_multiple,
    )
    output["aio-clock"] = _load_binary_file(
        path_to_folder / f"analog-clock_{index}.raw",
        dtype=ONIX_DATA_FORMAT["clock"],
        nchan=1,
        cut_if_not_multiple=cut_if_not_multiple,
    )
    if verbose:
        print(f"{num_ai_chan} AI channels found")

    digital_input = dict()
    digital_input["time"] = (
        np.fromfile(path_to_folder / f"digital-clock_{index}.raw", dtype=np.uint64)
        / meta["acq_clk_hz"]
    )
    digital_input["pins"] = np.fromfile(
        path_to_folder / f"digital-pins_{index}.raw", dtype=np.uint8
    )
    digital_input["buttons"] = np.fromfile(
        path_to_folder / f"digital-buttons_{index}.raw", dtype=np.uint16
    )
    digital_input["pins_b"] = (
        np.unpackbits(digital_input["pins"], bitorder="little")
        .reshape(-1, 8)
        .astype(bool)
    )
    digital_input["buttons_b"] = (
        np.unpackbits(digital_input["buttons"].astype(np.uint8), bitorder="little")
        .reshape(-1, 8)
        .astype(bool)
    )
    output["digital_input"] = digital_input
    if verbose:
        print(f"{len(digital_input['pins'])} digital input found")

    # Add clock info and port status
    output["output-clock"] = np.loadtxt(
        path_to_folder / f"output-clock_{index}.csv", dtype=int, delimiter=","
    )
    # Port status might be an empty file, remove userwarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        output["port-status"] = np.loadtxt(
            path_to_folder / f"port-status_{index}.csv", dtype=int, delimiter=","
        )

    # Read harp sync
    output["harpsync"] = pd.read_csv(
        path_to_folder / f"harp-sync_{index}.csv",
        dtype=int,
        delimiter=",",
        names=["onix_sample", "harp_timestamp"],
    )
    return output


def load_bno055(
    path_to_folder,
    timestamp=None,
    num_chans_euler=3,
    num_chans_gravity=3,
    num_chans_linear_accel=3,
    num_chans_quaternion=4,
    cut_if_not_multiple=False,
    ignore_wrong_timestamps=False,
):
    """Loads the IMU data in a memmap dictionary
    Args:
        path_to_folder (str or Path): the full path to the folder which contains the IMU output.
        timestamp (str or None): timestamp used in save name
        num_chans_euler (int): number of channels for the euler angles (default 3)
        num_chans_gravity (int): number of channels for the gravity vector (default 3)
        num_chans_linear_accel (int): number of channels for the linear acceleration (default 3)
        num_chans_quaternion (int): number of channels for the quaternion (default 4)
        cut_if_not_multiple (bool): if True, will cut the data if it is not a multiple
            of the number of channels if False, will load only if the data is a multiple
            of the number of channels. Default False.
        ignore_wrong_timestamps (bool): if True and timestamp is None, will keep all
            files with the prefix without looking at timestamps, if False, will raise an
            error if multiple timestamps are found. Default False.

    Returns:
        bno_out: a dictionary of memmap
    """

    num_chan_dict = dict(
        euler=num_chans_euler,
        gravity=num_chans_gravity,
        linear=num_chans_linear_accel,
        quaternion=num_chans_quaternion,
    )

    bno_files = _find_files(
        path_to_folder,
        timestamp,
        "bno055",
        ignore_wrong_timestamps=ignore_wrong_timestamps,
    )
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
        data = _load_binary_file(
            bno_file,
            dtype=np.double,
            nchan=num_chan_dict[what],
            order="F",
            cut_if_not_multiple=cut_if_not_multiple,
        )

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
    datatype and expects int16. This function copies the data to the new file changing
    the datatype.

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


def _find_files(folder, timestamp, prefix, ignore_wrong_timestamps=False):
    """Inner function to return list of files with filter_name and timestamp

    Args:
        folder(str or Path): path to the folder containing data
        timestamp (str or None): timestamp used in save name
        prefix (str): prefix filter
        ignore_wrong_timestamps (bool): if True and timestamp is None, will keep all
            files with the prefix without looking at timestamps, if False, will raise an
            error if multiple timestamps are found. Default False.

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
        if ignore_wrong_timestamps:
            return valid_files
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
