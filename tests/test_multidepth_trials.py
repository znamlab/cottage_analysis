import contextlib
import io

import numpy as np
import pandas as pd

from cottage_analysis.analysis.spheres import multidepth

DEPTHS = [5, 20, 80]
TRIALS = [(0, 4), (12, 16), (24, 28), (36, 40)]


def make_param_log(late_offset=None):
    """Param logs of 3 depth loggers, spheres shown during TRIALS.

    Args:
        late_offset (tuple): (depth, trial index, delay) to delay the offset of one
            trial in one logger.
    """
    rows = []
    times = np.arange(0, 46, 0.1)
    for depth in DEPTHS:
        on = np.zeros(len(times), dtype=bool)
        for itrial, (start, stop) in enumerate(TRIALS):
            if late_offset is not None and late_offset[:2] == (depth, itrial):
                stop += late_offset[2]
            on |= (times >= start) & (times < stop)
        rows.append(
            pd.DataFrame(
                {
                    "HarpTime": times,
                    "Radius": np.where(on, depth, -9999),
                    "logger_fname": f"NewParams_{depth}cm.csv",
                }
            )
        )
    return pd.concat(rows).sort_values("HarpTime", kind="stable").reset_index(drop=True)


def find_trials(param_log):
    with contextlib.redirect_stdout(io.StringIO()):
        trial_on_off, _ = multidepth.find_trial_times(param_log)
    return np.round(trial_on_off.T, 1)


def test_all_trials_found():
    np.testing.assert_allclose(find_trials(make_param_log()), TRIALS)


def test_trial_with_bad_offset_is_dropped_not_merged():
    # the 20 cm logger turns off 3 s late in the second trial: that trial fails the
    # jitter check and must be excluded, not merged with the third trial
    trials = find_trials(make_param_log(late_offset=(20, 1, 3)))
    np.testing.assert_allclose(trials, [TRIALS[0], TRIALS[2], TRIALS[3]])
