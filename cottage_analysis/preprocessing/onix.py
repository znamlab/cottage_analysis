import numpy as np
from cottage_analysis.io_module import onix

DIGITAL_INPUTS = dict(DI0="fm_cam_trig", DI1="oni_clock_di", DI2="hf_cam_trig")
ANALOG_INPUTS = ["none", "wehrcam", "photodiode", "none"]
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


def preprocess_onix_recording(
    data, harp_message, breakout_di_names=None, debounce_window=1000
):
    """Preprocess the ONIX recording data.

    Args:
        data (dict): The ONIX recording data, as return by io_module.onix.load_recording
        harp_message (dict): The harp message, as returned by io_module.harp.load_harp
        breakout_di_names (dict, optional): A dictionary mapping the breakout digital
        debounce_window (int, optional): Window to debounce signal in samples. Defaults
            to 1000.

    Returns:
        dict: The preprocessed ONIX recording data. (same as input)
    """
    data["breakout_data"] = clean_breakout(
        data["breakout_data"], breakout_di_names, debounce_window=debounce_window
    )
    h2o, o2h = sync_harp2onix(
        harp_message, data["breakout_data"]["digital_inputs"]["oni_clock_di"]
    )
    data["harp2onix"], data["onix2harp"] = h2o, o2h
    return data


def sync_harp2onix(harp_message, oni_clock_di):
    """Synchronise harp to onix using clock digital input

    Args:
        harp_message: output of load_harp

    Returns:
        harp_message: same as input, with two new fields: 'analog_time_onix' and
            'digital_time_onix'
        harp2onix: conversion function
    """

    # find when onix clock sends a heartbeat
    is_onix_heartbeat = np.diff(np.hstack([0, harp_message["onix_clock"]])) == 1
    heart_harp = harp_message["digital_time"][is_onix_heartbeat]
    is_onix_heartbeat = np.diff(np.hstack([0, oni_clock_di[1, :]])) == 1
    heart_oni = oni_clock_di[0, is_onix_heartbeat]

    # check that I have the same number of heartbeats. If not, warn and cut the longest
    # one if the difference is less than 1% of the total number of heartbeats
    nharp = len(heart_harp)
    noni = len(heart_oni)
    if nharp != noni:
        if nharp < noni and nharp / noni > 0.99:
            print("Cutting %d heartbeats from onix clock" % (noni - nharp))
            heart_oni = heart_oni[:nharp]
        elif nharp > noni and noni / nharp > 0.99:
            print("Cutting %d heartbeats from harp clock" % (nharp - noni))
            heart_harp = heart_harp[:noni]
        else:
            raise ValueError(
                "The number of heartbeats in the harp and onix clock do not match. "
                "This is likely due to a recording error. "
                "The number of harp heartbeats is %d and the number of onix heartbeats is %d"
                % (nharp, noni)
            )

    harp_frq = 1 / np.median(np.diff(heart_harp))
    onix_frq = 1 / np.median(np.diff(heart_oni)) * onix.ONIX_SAMPLING
    print(f"Harp clock frequency: {harp_frq:.2f}Hz")
    print(f"Onix clock frequency: {onix_frq:.2f}Hz")
    assert (
        np.abs(harp_frq - onix_frq) < 0.5
    ), "Harp and Onix clock frequencies differ by more than 0.5Hz"

    def harp2onix(data):
        """Convert harp timestamp in onix time"""
        return np.interp(data, heart_harp, heart_oni.astype(float)).astype("int64")

    def onix2harp(data):
        """Convert onix timestamp in harp time"""
        return np.interp(data, heart_oni.astype(float), heart_harp)

    return harp2onix, onix2harp


def clean_breakout(breakout_data, breakout_di_names=None, debounce_window=1000):
    """Clean the breakout data.

    Args:
        breakout_data (dict): The breakout data, as returned by `load_breakout`.
        breakout_di_names (dict): A dictionary mapping the breakout digital input names
            to the names used in the data.
        debounce_window (int, optional): Window to debounce signal in samples. Defaults
            to 1000.

    Returns:
        dict: The cleaned breakout data.
    """
    breakout_data["digital_inputs"] = clean_di(
        breakout_data["dio"], debounce_window=debounce_window
    )

    if breakout_di_names is None:
        breakout_di_names = DIGITAL_INPUTS.copy()

    for di in list(breakout_data["digital_inputs"].keys()):
        if di in breakout_di_names:
            breakout_data["digital_inputs"][breakout_di_names[di]] = breakout_data[
                "digital_inputs"
            ].pop(di)

    n_clock = breakout_data["aio-clock"].shape[0]
    n_data = breakout_data["aio"].shape[1]
    if n_clock != n_data:
        # There can be a small difference at the end, crop to the shortest if we loose
        # less than 1% of the samples by doing so, otherwise raise an error
        if n_clock < n_data and n_clock / n_data > 0.99:
            print(f"Cutting {n_data - n_clock} samples from the end of the analog data")
            breakout_data["aio"] = breakout_data["aio"][:, :n_clock]
        elif n_clock > n_data and n_data / n_clock > 0.99:
            print(f"Cutting {n_clock - n_data} samples from the end of the ai clock")
            breakout_data["aio-clock"] = breakout_data["aio-clock"][:n_data]
        else:
            raise ValueError(
                "The number of samples in the clock and data do not match. "
                "This is likely due to a recording error. "
                "The number of clock samples is %d and the number of data samples is %d"
                % (n_clock, n_data)
            )

    return breakout_data


def clean_di(dio, debounce_window=1000):
    """Clean the digital input signals.

    Args:
        dio (dict): The digital input signals, as returned by `load_breakout`.
        debounce_window (int, optional): Window to debounce signal in samples. Defaults
            to 1000.

    Returns:
        dict: The cleaned digital input signals.
    """
    dis = {}
    full_clock = dio["Clock"]
    for i in range(8):
        values, clock = debounce(dio["DI%d" % i], full_clock, debounce_window)
        dis["DI%d" % i] = np.vstack([clock, values])
    return dis


def debounce(values, clock, window):
    """Debounce a digital input signal.

    Args:
        values (np.ndarray): The digital input values.
        clock (np.ndarray): The clock values.
        window (int): The debounce window in samples.

    Returns:
        np.ndarray: The debounced values.
        np.ndarray: The debounced clock.
    """
    values = np.array(values, dtype=bool)
    clock = np.array(clock)
    val_change = np.zeros(values.shape, dtype=bool)
    val_change[1:] = np.diff(values) != 0
    if not any(val_change):
        return np.array([values[0]]), np.array([clock[0]])
    values = values[val_change]
    clock = clock[val_change]
    valid_transition = np.hstack([True, np.diff(clock) > window])
    return values[valid_transition], clock[valid_transition]
