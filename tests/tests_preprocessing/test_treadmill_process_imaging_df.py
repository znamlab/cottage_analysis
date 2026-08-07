"""Unit tests for `cottage_analysis.analysis.treadmill.process_imaging_df`.

These are self-contained: they build a minimal synthetic `imaging_df` (a step
`MotorSps` command plus a controlled `RS` trace) rather than depending on
flexilims/real session data, so they run offline.
"""

import numpy as np
import pandas as pd
import pytest

from cottage_analysis.analysis import treadmill
from cottage_analysis.analysis.treadmill import (
    ACTUAL_MOTOR_SPEED,
    CIRCUMFERENCE,
    MICROSTEPPING,
    STEPS_PER_REV,
    _match_ramp_template,
    process_imaging_df,
    sps2speed,
)


def _sps_for_key(key):
    """Inverse of `sps2speed`: the raw `MotorSps` value whose calibrated speed is
    `ACTUAL_MOTOR_SPEED[key]`, chosen so `round(sps2speed(sps)) == key` exactly."""
    return key * STEPS_PER_REV / (MICROSTEPPING * CIRCUMFERENCE)


def _build_imaging_df(segments, frame_rate=15.0, depth=0.2):
    """Build a synthetic `imaging_df` from a list of segments.

    Args:
        segments: list of `(duration_s, motor_sps, rs_fn)`. `rs_fn` is called with
            the segment-local time array (seconds, starting at 0) and must return
            RS **in cm/s**; ignored (RS forced to 0) when `motor_sps == 0`.
        frame_rate: sampling rate of the synthetic `imaging_harptime` grid, Hz.
        depth: constant `depth` column (m), only needed so `expected_optic_flow`
            doesn't divide by zero.

    Returns:
        (pd.DataFrame, list[dict]): the imaging_df, and one dict per motor-on
        segment with `command_start_idx` (frame index of the first on-sample),
        `duration_frames`, and `motor_speed` (calibrated cm/s).
    """
    dt = 1.0 / frame_rate
    harptime_chunks, motor_chunks, rs_chunks = [], [], []
    trials = []
    frame_offset = 0
    for duration_s, sps, rs_fn in segments:
        n = max(1, int(round(duration_s / dt)))
        t_local = np.arange(n) * dt
        harptime_chunks.append(frame_offset * dt + t_local)
        motor_chunks.append(np.full(n, sps, dtype=float))
        if sps > 0:
            trials.append(
                dict(
                    command_start_idx=frame_offset,
                    duration_frames=n,
                    motor_speed=ACTUAL_MOTOR_SPEED[int(round(sps2speed(sps)))],
                )
            )
            rs_chunks.append(rs_fn(t_local))
        else:
            rs_chunks.append(np.zeros(n))
        frame_offset += n

    harptime = np.concatenate(harptime_chunks)
    motor = np.concatenate(motor_chunks)
    rs_cm = np.concatenate(rs_chunks)
    df = pd.DataFrame(
        {
            "imaging_harptime": harptime,
            "MotorSps": motor,
            "RS": rs_cm / 100.0,  # cm/s -> m/s, matching real imaging_df.RS units
            "depth": np.full(len(harptime), depth),
        }
    )
    return df, trials


def _legacy_new_start_index(trial, acceleration_time, margin, frame_rate):
    """The original hardcoded formula, used as the regression oracle."""
    acc_frames = int((acceleration_time * trial["motor_speed"] + margin) * frame_rate)
    return trial["command_start_idx"] + acc_frames


def _flat_rs(v_cm_s):
    """RS held at a constant value for a whole segment (irrelevant to method='model')."""
    return lambda t_local: np.full(len(t_local), v_cm_s)


def _ramp_rs(true_shift, acceleration, plateau_speed):
    """A clean trapezoidal ramp with a known physical onset, for the plateau tests."""
    return lambda t_local: np.clip(acceleration * (t_local - true_shift), 0.0, plateau_speed)


FRAME_RATE = 15.0
ACCELERATION_TIME = 0.13


def test_model_method_matches_legacy_formula_default_margin():
    df, trials = _build_imaging_df(
        [
            (2.0, 0, None),
            (6.0, _sps_for_key(16), _flat_rs(0.0)),  # 15.25 cm/s
            (2.0, 0, None),
            (4.0, _sps_for_key(32), _flat_rs(0.0)),  # 30.5 cm/s
            (2.0, 0, None),
        ],
        frame_rate=FRAME_RATE,
    )
    out = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="model"
    )
    starts = np.flatnonzero(out.is_trial_start.values)
    expected = [
        _legacy_new_start_index(t, ACCELERATION_TIME, 0.5, FRAME_RATE) for t in trials
    ]
    assert list(starts) == expected


def test_model_method_matches_legacy_formula_custom_margin():
    df, trials = _build_imaging_df(
        [
            (2.0, 0, None),
            (6.0, _sps_for_key(8), _flat_rs(0.0)),  # 7.625 cm/s
            (2.0, 0, None),
        ],
        frame_rate=FRAME_RATE,
    )
    margin = 1.2
    out = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="model", margin=margin
    )
    starts = np.flatnonzero(out.is_trial_start.values)
    expected = [
        _legacy_new_start_index(t, ACCELERATION_TIME, margin, FRAME_RATE) for t in trials
    ]
    assert list(starts) == expected


def test_plateau_method_recovers_synthetic_onset():
    v = ACTUAL_MOTOR_SPEED[16]  # 15.25 cm/s
    acceleration = 1.0 / ACCELERATION_TIME
    true_shift = 0.3
    duration = 6.0
    margin = 0.5

    df, trials = _build_imaging_df(
        [
            (2.0, 0, None),
            (duration, _sps_for_key(16), _ramp_rs(true_shift, acceleration, v)),
            (2.0, 0, None),
        ],
        frame_rate=FRAME_RATE,
    )
    out = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="plateau", margin=margin
    )
    starts = np.flatnonzero(out.is_trial_start.values)
    assert len(starts) == 1

    ramp = v / acceleration
    expected_offset = round((true_shift + ramp + margin) * FRAME_RATE)
    expected_idx = trials[0]["command_start_idx"] + expected_offset
    assert abs(starts[0] - expected_idx) <= 2  # within ~2 frames


def test_plateau_method_recovers_synthetic_onset_with_noise():
    v = ACTUAL_MOTOR_SPEED[32]  # 30.5 cm/s
    acceleration = 1.0 / ACCELERATION_TIME
    true_shift = 0.4
    duration = 6.0
    margin = 0.5
    rng = np.random.default_rng(seed=0)

    def noisy_ramp(t_local):
        clean = np.clip(acceleration * (t_local - true_shift), 0.0, v)
        return clean + rng.normal(scale=0.3 * v / 30.5, size=len(t_local))

    df, trials = _build_imaging_df(
        [(2.0, 0, None), (duration, _sps_for_key(32), noisy_ramp), (2.0, 0, None)],
        frame_rate=FRAME_RATE,
    )
    out = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="plateau", margin=margin
    )
    starts = np.flatnonzero(out.is_trial_start.values)
    assert len(starts) == 1

    ramp = v / acceleration
    expected_offset = round((true_shift + ramp + margin) * FRAME_RATE)
    expected_idx = trials[0]["command_start_idx"] + expected_offset
    assert abs(starts[0] - expected_idx) <= 4  # a bit looser with noise


def test_plateau_method_falls_back_to_model_when_detector_returns_nan():
    v = ACTUAL_MOTOR_SPEED[16]  # 15.25 cm/s
    acceleration = 1.0 / ACCELERATION_TIME
    ramp = v / acceleration
    # duration too short to leave room for ramp + MIN_PLATEAU_S regardless of shift
    duration = ramp + treadmill._PLATEAU_MIN_PLATEAU_S - 0.5
    margin = 0.5

    df, trials = _build_imaging_df(
        [
            (2.0, 0, None),
            (duration, _sps_for_key(16), _ramp_rs(0.0, acceleration, v)),
            (2.0, 0, None),
        ],
        frame_rate=FRAME_RATE,
    )
    out_plateau = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="plateau", margin=margin
    )
    out_model = process_imaging_df(
        df.copy(), acceleration_time=ACCELERATION_TIME, method="model", margin=margin
    )
    starts_plateau = np.flatnonzero(out_plateau.is_trial_start.values)
    starts_model = np.flatnonzero(out_model.is_trial_start.values)
    assert list(starts_plateau) == list(starts_model)


def test_match_ramp_template_returns_nan_for_insufficient_window():
    v = ACTUAL_MOTOR_SPEED[16]
    acceleration = 1.0 / ACCELERATION_TIME
    ramp = v / acceleration
    duration = ramp + treadmill._PLATEAU_MIN_PLATEAU_S - 0.5
    t = np.linspace(0, duration, 20)
    rs = np.clip(acceleration * t, 0, v)
    t_plateau, shift, cost = _match_ramp_template(t, rs, v, duration, acceleration)
    assert np.isnan(t_plateau) and np.isnan(shift) and np.isnan(cost)


def test_rs_unit_conversion_is_required():
    """`_match_ramp_template` expects `rs` in cm/s. Feeding it m/s-scale data (i.e.
    forgetting the `* 100.0` conversion done in `process_imaging_df`) should fail to
    recover the true onset, pinning that conversion as required, not cosmetic."""
    v = ACTUAL_MOTOR_SPEED[16]
    acceleration = 1.0 / ACCELERATION_TIME
    true_shift = 0.3
    duration = 6.0
    t = np.arange(0, duration, 1 / FRAME_RATE)
    rs_cm = np.clip(acceleration * (t - true_shift), 0.0, v)

    t_plateau_correct, shift_correct, _ = _match_ramp_template(
        t, rs_cm, v, duration, acceleration
    )
    t_plateau_wrong, shift_wrong, _ = _match_ramp_template(
        t, rs_cm / 100.0, v, duration, acceleration  # wrong: still in m/s scale
    )
    assert abs(shift_correct - true_shift) < 0.05
    assert abs(shift_wrong - shift_correct) > 0.2


def test_invalid_method_raises():
    df, _ = _build_imaging_df(
        [(2.0, 0, None), (4.0, _sps_for_key(16), _flat_rs(0.0)), (2.0, 0, None)],
        frame_rate=FRAME_RATE,
    )
    with pytest.raises(ValueError):
        process_imaging_df(df, acceleration_time=ACCELERATION_TIME, method="bogus")


def test_trial_duration_and_acceleration_time_mutually_exclusive_still_enforced():
    df, _ = _build_imaging_df(
        [(2.0, 0, None), (4.0, _sps_for_key(16), _flat_rs(0.0)), (2.0, 0, None)],
        frame_rate=FRAME_RATE,
    )
    with pytest.raises(ValueError):
        process_imaging_df(df, trial_duration=2.0, acceleration_time=ACCELERATION_TIME)
