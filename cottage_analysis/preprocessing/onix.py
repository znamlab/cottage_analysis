import numpy as np

BREAKOUT_DIGITAL_INPUTS = dict(DI0="fm_cam_trig", DI1="oni_clock_di", DI2="hf_cam_trig")


def preprocess_onix_recording(data, breakout_di_names=None, debounce_window=1000):
    """Preprocess the ONIX recording data.

    Args:
        data (dict): The ONIX recording data, as return by io_module.onix.load_recording
        breakout_di_names (dict, optional): A dictionary mapping the breakout digital
        debounce_window (int, optional): Window to debounce signal in samples. Defaults
            to 1000.

    Returns:
        dict: The preprocessed ONIX recording data. (same as input)
    """
    if breakout_di_names is None:
        breakout_di_names = BREAKOUT_DIGITAL_INPUTS
    dis = clean_di(data["breakout_data"]["dio"], debounce_window=debounce_window)
    for di in list(dis.keys()):
        if di in breakout_di_names:
            dis[breakout_di_names[di]] = dis.pop(di)
    data["breakout_data"]["digital_inputs"] = dis
    return data


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
