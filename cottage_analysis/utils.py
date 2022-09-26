"""
Common utility functions

If it applies to regularly sampled continuous data, see continuous_data_analysis
If it applies to point processes or series of time, see time_series_analysis
"""

import numpy as np


def flaten_list(list_of_arrays):
    """Return a single array from list of arrays

    Usefull mostly if you have lot of arrays with different length
    (typically a crosscorrelogram)"""
    if not list_of_arrays:
        return np.array([])
    ntot = sum(map(np.size, list_of_arrays))
    out = np.array(np.zeros(ntot), dtype=list_of_arrays[0].dtype)
    n = 0
    for i, j in enumerate(map(np.size, list_of_arrays)):
        out[n:n + j] = list_of_arrays[i]
        n += j
    return out