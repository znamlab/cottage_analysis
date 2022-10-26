"""
Module to find frames based on photodiode flicker
"""
import time

import numpy as np
import pandas as pd
import scipy.signal as scsi
import matplotlib.pyplot as plt
from cottage_analysis.utilities import continuous_data_analysis as cda
from cottage_analysis.utilities.time_series_analysis import searchclosest


def sync_by_correlation(frame_log, photodiode_time, photodiode_signal,
                        time_column='HarpTime', sequence_column='PhotoQuadColor',
                        num_frame_to_corr=5, maxlag=50e-3, expected_lag=15e-3,
                        frame_rate=144, correlation_threshold=0.9,
                        relative_corr_thres=0.03, minimum_lag=5e-3, do_plot=False,
                        verbose=True, debug=False):
    """Find best shift to synchronise photodiode with ideal sequence

    This will generate an idealised photodiode trace based on the `switch_time` and the
    switch `sequence`. This trace will be synced by crosscorrelation

    Args:
        frame_log (float): The logger from bonsai with RenderFrame values
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        time_column (str): Name of the column in `frame_log` containing time. Must
                           match photodiode_time (Default to HarpTime)
        sequence_column (str): Name of the column in `frame_log` containing sequence
                               information (Default to 'PhotoQuadColor')
        num_frame_to_corr (int): number of frame before and after frame_time to keep
                                 for correlation
        maxlag (float): Maximum lag tested (in s, centered on expected_lag).
        expected_lag (float): expected lag (in s) to center search
        frame_rate (float): Frame rate in Hz
        correlation_threshold (float): threshold on the pearson correlation. Anything
                                       below is considered a failure to fit
        relative_corr_thres (float): When multiple correlations are above threshold,
                                     will consider only among those that less than
                                     `relative_corr_thres` away from max corr
        minimum_lag (float): Minimum possible lag. Anything below is considered a
                             failure to fit
        do_plot (bool): If True generate some quality measure plots during run and
                        return the figure handles
        verbose (bool): Print progress and general info.
        debug (bool): False by default. If True, returns a dict with intermediary results

    Returns:
        frames_df (pd.DataFrame): dataframe with a line per detected frame
        extra_out (dict): A dictionary containing figures if `do_plot` is True,
                          debug information if debug is True. Empty if both are False
    """

    # First step: Frame detection
    pd_sampling = 1/np.mean(np.diff(photodiode_time))
    out = detect_frame_onset(photodiode=photodiode_signal,
                             frame_rate=frame_rate,
                             photodiode_sampling=pd_sampling,
                             highcut=frame_rate * 3,
                             debug=debug or do_plot)
    if debug or do_plot:
        frame_borders, peak_index, db_dict = out
    else:
        frame_borders, peak_index = out
    # cut frame detected before the recording started
    t0 = frame_log[time_column].iloc[0]
    to_cut = photodiode_time[frame_borders].searchsorted(t0)
    frame_borders = frame_borders[to_cut:]
    peak_index = peak_index[to_cut:]

    # Format the results in a nicer dataframe
    frame_skip = np.diff(frame_borders) > pd_sampling / frame_rate * 1.5
    frames_df = pd.DataFrame(dict(onset_sample=frame_borders[:-1],
                                  offset_sample=frame_borders[1:],
                                  peak_sample=peak_index,
                                  include_skip=frame_skip))
    frames_df['onset_time'] = photodiode_time[frames_df.onset_sample]
    frames_df['offset_time'] = photodiode_time[frames_df.offset_sample]
    frames_df['peak_time'] = photodiode_time[frames_df.peak_sample]
    if verbose:
        print('Found %d frames out of %d Render'
              ' (%d%%, %d dropped)' % (len(frames_df), len(frame_log),
                                       len(frames_df) / len(frame_log) * 100,
                                       len(frame_log) - len(frames_df)))

    if do_plot:
        plot_window = np.array([-7.5, 7.5]) / frame_rate * pd_sampling
        figs = plot_frame_detection_report(border_index=frame_borders,
                                           peak_index=peak_index,
                                           debug_dict=db_dict,
                                           num_examples=1,
                                           plot_window=plot_window,
                                           photodiode=photodiode_signal,
                                           frame_rate=frame_rate,
                                           photodiode_sampling=pd_sampling,
                                           highcut=frame_rate * 3)
        fig_dict = dict(frame_dection=figs)

    # Second step: cross correlation
    normed_pd = np.array(photodiode_signal, dtype=float)
    normed_pd -= np.quantile(normed_pd, 0.01)
    normed_pd /= np.quantile(normed_pd, 0.99)
    frame_onsets = frames_df['onset_sample'].values
    maxlag_samples = int(np.round(maxlag * pd_sampling))  # make it into samples
    expected_lag_samples = int(np.round(expected_lag * pd_sampling))  # make it into samples
    out = _crosscorr_befcentaft(frame_onsets,
                                photodiode_time=photodiode_time,
                                photodiode_signal=normed_pd,
                                switch_time=frame_log[time_column].values,
                                sequence=frame_log[sequence_column].values,
                                expected_lag=expected_lag_samples,
                                maxlag=maxlag_samples,
                                num_frame_to_corr=num_frame_to_corr,
                                frame_rate=frame_rate,
                                verbose=verbose,
                                debug=debug)
    if debug:
        cc_dict, lags, db = out
        db_dict.update(db)
        db_dict['normed_pd'] = normed_pd
        db_dict['lags_sample'] = lags
        db_dict['cc_dict'] = cc_dict
    else:
        cc_dict, lags = out
    # add that to the dataframe
    align = dict(bef='onset_time', center='peak_time', aft='offset_time')
    func = dict(bef=searchclosest, center=np.searchsorted, aft=searchclosest)
    shift = dict(bef=0, center=-1, aft=-1)
    for iw, which in enumerate(['bef', 'center', 'aft']):
        frames_df['lag_%s' % which] = lags[cc_dict[which].argmax(axis=1)] / pd_sampling
        frames_df['peak_corr_%s' % which] = cc_dict[which].max(axis=1)
        # find the closest frame, looking at onset for before, peak for center and
        # offset for after.
        cl = func[which](frame_log[time_column].values,
                         (frames_df[align[which]] - frames_df['lag_%s' % which]).values)
        # To lag each element from frames_df by a different lag, I subtract the lag
        # instead of adding to frame_log
        # To have the proper number of element I search frame_log in frames_df instead
        # of the converse. That means that I get the index of frame_log that is >=
        # frames_df
        cl += shift[which]
        cl = np.clip(cl, 0, len(frame_log) - 1)
        frames_df['closest_frame_%s' % which] = cl
        frames_df['quadcolor_%s' % which] = frame_log.iloc[cl][sequence_column].values

    # also add photodiode value at peak
    frames_df['photodiode'] = normed_pd[peak_index]
    # use this measure to find frame "jumps", where there is no alternation
    diff_sign = np.sign(np.diff(frames_df.photodiode))
    jumps = diff_sign[1:] == diff_sign[:-1]
    frames_df['is_jump'] = np.hstack([0, jumps, 0])

    # Now attempt the matching
    frames_df = _match_fit_to_logger(frames_df, frame_log,
                                     correlation_threshold=correlation_threshold,
                                     relative_corr_thres=relative_corr_thres,
                                     minimum_lag=minimum_lag,
                                     clean_df=not debug,
                                     verbose=verbose)
    extra_out = {}
    if do_plot:
        extra_out['figures'] = fig_dict
    if debug:
        extra_out['debug_info'] = db_dict
    return frames_df, extra_out


def plot_on_frame_check(frame, frames_df, frame_log, db_dict):

    fs = frames_df.loc[frame]
    win = np.array([-20, 20])
    t0 = fs.onset_time
    fig = plt.figure(figsize=(5, 6))
    d = dict(bef='onset_time', center='peak_time', aft='offset_time')
    iax=1
    for w in ['bef', 'center', 'aft']:
        plt.subplot(3, 1, iax)
        i = fs['closest_frame_%s'%w] + np.arange(-3, 4, dtype=int)
        i0 =fs['closest_frame_%s'%w]
        plt.axvspan(fs.onset_time-t0, fs.offset_time-t0, color='purple', alpha=0.5)
        plt.axvline(fs[d[w]]-t0, color='k', ls='--')
        plt.plot(photodiode_time[slice(*win + fs.peak_sample)]-t0,
                 db_dict['normed_pd'][slice(*win + fs.peak_sample)])
        plt.plot(photodiode_time[slice(*win + fs.peak_sample)] -t0+ fs['lag_%s'%w],
                 db_dict['ideal_pd'][slice(*win + fs.peak_sample)])
        plt.plot(frame_log.loc[i, 'HarpTime'] + fs['lag_%s'%w]-t0,
                 frame_log.loc[i, 'PhotoQuadColor'],
                 drawstyle='steps-post')
        plt.plot(frame_log.loc[[i0, i0+1], 'HarpTime'] + fs['lag_%s'%w]-t0,
                 np.zeros(2) + frame_log.loc[i0, 'PhotoQuadColor'], 'k',
                 lw=4)
        plt.axvline(frame_log.loc[i0, 'HarpTime'] + fs['lag_%s'%w]-t0,
                    color='k', ls=':')
        plt.ylabel(w)
        iax+=1
    plt.show()


def detect_frame_onset(photodiode, frame_rate=144, photodiode_sampling=1000,
                       highcut=400, debug=False):
    """Detect frames from photodiode signal

    Simple wrapper around `scipy.signal.find_peaks` to detect frame borders from
    photodiode signal.

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float): expected frame rate, peaks occuring faster than two frame
                            rate will be ignored
        photodiode_sampling (float): Sampling rate of the photodiode signal
        highcut (float): If not None, use low pass filter cutting components
                         above `highcut` hertz
        debug (bool): False by default. If True, returns a dict with intermediary results

    Returns:
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True
    """
    if highcut is not None:
        fpd = cda.filter(photodiode, highcut=frame_rate * 3, sampling=photodiode_sampling,
                         design='bessel')
        absdiff = np.abs(np.diff(fpd))
    else:
        fpd = None
        absdiff = np.abs(np.diff(photodiode))
    absdiff -= np.quantile(absdiff, 0.01)
    absdiff /= np.quantile(absdiff, 0.99)
    dst = int(1 / (frame_rate * 2) * photodiode_sampling)
    frame_border, _ = scsi.find_peaks(absdiff, distance=dst, height=0.05)

    # When two local peaks in the same frame have exactly the same amplitude,
    # find_peaks doesn't know which one to pick and keeps both. Let's keep only the last
    plateau = np.where((np.diff(frame_border) <= dst) &
                       (np.diff(absdiff[frame_border]) == 0))[0]
    valid = np.ones(len(frame_border), dtype=bool)
    valid[plateau] = False
    borders = np.array(frame_border[valid])

    # Now find the peak of the frame, it is len(frame_border) - 1
    peaks = np.array([absdiff[b:e].argmin() + b
                      for b, e in zip(frame_border[:-1], frame_border[1:])], dtype=int)
    if debug:
        debug_dict = dict(plateau=plateau, all_pks=frame_border, distance=dst,
                          filtered_trace=fpd, diff_trace=absdiff)
        return borders, peaks, debug_dict
    return borders, peaks


def plot_frame_detection_report(photodiode, frame_rate=144, photodiode_sampling=1000,
                                highcut=400, plot_window=(-50, 50), num_examples=1,
                                border_index=None, peak_index=None, debug_dict=None):
    """Detect frames and generate a few debuging figures

    If border_index, peak_index or debug_dict is None, frame detection will be
    performed first, otherwise the results will be used directly.
    This will select `num_examples` frames randomly and plot `plot_window` samples
    around them. Another `num_examples` of frames with a frame drop will be selected
    and ploted the same way

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float): expected frame rate, peaks occuring faster than two frame
                            rate will be ignored
        photodiode_sampling (float): Sampling rate of the photodiode signal
        highcut (float): If not None, use low pass filter cutting components
                         above `highcut` hertz
        plot_window ([int, int]): limit of the window to plot around each example frame
        num_examples (int): number of figures randomly selected and with frame drop to
                            plot (len(fig) == num_examples * 2)
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True


    Returns:
        figs (list): a list of figure handles
    """
    if (border_index is None) or (debug_dict is None):
        border_index, debug_dict = detect_frame_onset(photodiode, frame_rate,
                                                      photodiode_sampling, highcut,
                                                      debug=True)
    rng = np.random.default_rng(42)
    skip = np.diff(border_index) > photodiode_sampling / frame_rate * 1.5
    example_frames = np.hstack([border_index[rng.integers(len(border_index),
                                                          size=num_examples)],
                                border_index[:-1][skip][rng.integers(np.sum(skip),
                                                                     size=num_examples)]])
    figs = []
    w = np.array(plot_window, dtype=int)
    i = np.arange(len(photodiode))
    vlines_kwargs = [dict(color='Grey', alpha=0.2, ls='--', lw=0.5),
                     dict(color='k', alpha=0.5, ls='-')]
    dot_kwargs = [dict(color='Grey', alpha=0.2, marker='.', s=5),
                  dict(color='darkred', alpha=0.5, marker='o',  s=50,
                       ec='None')]
    for f in example_frames:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(debug_dict['filtered_trace'][slice(*w+f)], label='filtered')
        ax.plot(photodiode[slice(*w + f)], label='raw')
        ax.legend(loc='upper right')
        for iw, which_pk in enumerate([debug_dict['all_pks'], border_index]):
            v = (which_pk > w[0] + f) & (which_pk < w[1] + f)
            for d in i[which_pk[v]] - (w[0]+f):
                ax.axvline(d, **vlines_kwargs[iw])
        v = peak_index[(peak_index > w[0] + f) & (peak_index < w[1] + f)]
        ax.scatter(i[v] - (w[0] + f), photodiode[v], **dot_kwargs[1])
        ax.set_title('Frame at sample %d' % f)
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(debug_dict['diff_trace'][slice(*w+f)])
        for iw, which_pk in enumerate([debug_dict['all_pks'], border_index]):
            v = which_pk[(which_pk > w[0] + f) & (which_pk < w[1] + f)]
            ax.scatter(i[v] - (w[0] + f), debug_dict['diff_trace'][v], **dot_kwargs[iw])
        ax.set_ylabel('Abs diff trace')
        ax.set_xlabel('Sample')
        figs.append(fig)
    return figs


def ideal_photodiode(time_base, switch_time, sequence, highcut=150):
    """Make an idealise photodiode trace from sequence

    The photodiode is imaging a quad that changes color every frame but the photodiode
    filter and the pixel response time make this alternation smooth and not step-wise.
    This function attempts to filter the raw sequence to re-create an ideal version of
    what the photodiode signal should look like


    Args:
        time_base (np.array): Time of the real photodiode signal, will be the shape of
                              the output
        switch_time (np.array): Time points at which the photodiode changes colour
        sequence (np.array): Values of the photodiode for each of the switch time
        highcut (float): Frequency for low pass filter

    Returns:
        perfect_sequence (np.array): Continuous version of photodiode_value (same size as
                                     time_base)
        fake_photodiode (np.array): Filtered version of perfect_sequence
    """
    sampling = 1/np.mean(np.diff(time_base))
    change_indices = time_base.searchsorted(switch_time)
    perfect_sequence = np.zeros_like(time_base)
    for i, v in enumerate(sequence[:-1]):
        perfect_sequence[change_indices[i]: change_indices[i+1]] = v

    freq = highcut / sampling
    sos = scsi.butter(N=1, Wn=freq, btype='lowpass', output='sos')
    fake_photodiode = scsi.sosfilt(sos, perfect_sequence)
    return perfect_sequence, fake_photodiode


def _crosscorr_befcentaft(frame_onsets, photodiode_time, photodiode_signal, switch_time,
                          sequence, expected_lag, maxlag, num_frame_to_corr, frame_rate,
                          verbose=True, debug=False):
    """Run three crosscorrelations before, centered on and after each frame time

    Inner function of `sync_by_correlation`.

    Args:
        frame_onsets (np.array): sample of photodiode signal at which each frame starts
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        switch_time (np.array): Time of all changes of photodiode quad colour
        sequence (np.array): Value of the quad colour after each switch.
        expected_lag (int): expected lag (in samples) to center search
        maxlag (int): Maximum lag tested (in samples, centered on expected_lag).
        num_frame_to_corr (int): number of frame around frame_time to keep for correlation
        frame_rate (float): Frame rate in Hz
        verbose (bool): Print time taken. Default True
        debug (bool): Return intermediary results Default False

    Returns:
        cc_mat (np.array): a (3 x len(frame_onsets) x len(lags)) array of correlation
                           coefficients
        lags (np.array): lag in samples
        db_dict (dict): only if debug=True. Dictionnary with intermediary results
    """
    photodiode_sampling = 1/np.mean(np.diff(photodiode_time))
    seq_trace, ideal_pd = ideal_photodiode(time_base=photodiode_time,
                                           switch_time=switch_time,
                                           sequence=sequence)

    window = [np.array([-1, 1]) * maxlag + np.array(w * num_frame_to_corr / frame_rate
                                                    * photodiode_sampling, dtype='int')
              for w in [np.array([-1, 0]), np.array([-0.5, 0.5]), np.array([0, 1])]]
    # for bef window, we add 1 frame to have the current frame included
    window[0] += int(1/frame_rate * photodiode_sampling)
    if verbose:
        start = time.time()
        print('Starting crosscorrelation', flush=True)
    cc_mat = np.zeros((len(window), len(frame_onsets), maxlag * 2)) + np.nan
    for iframe, foi in enumerate(frame_onsets):
        for iw, win in enumerate(window):
            if (win[0] + foi) < 0 and verbose:
                print('Frame %d at sample %d is too close from start of recording' % (
                    iframe, foi))
                continue
            elif (win[1] + foi) > (len(photodiode_signal) - expected_lag) and verbose:
                print('Frame %d at sample %d is too close from end of recording' % (
                    iframe, foi))
                continue
            corr, lags = cda.crosscorrelation(photodiode_signal[slice(*win + foi)],
                                              ideal_pd[slice(*win + foi - expected_lag)],
                                              maxlag=maxlag,
                                              expected_lag=0,
                                              normalisation='pearson')
            cc_mat[iw, iframe] = corr
    lags += expected_lag
    if verbose:
        end = time.time()
        print('done (%d s)' % (end - start), flush=True)
    cc_dict = {l: cc_mat[i] for i, l in enumerate(['bef', 'center', 'aft'])}
    if debug:
        db_dict = dict(window=window, seq_trace=seq_trace, ideal_pd=ideal_pd)
        return cc_dict, lags, db_dict
    return cc_dict, lags


def _match_fit_to_logger(frames_df, frame_log, correlation_threshold=0.8,
                         relative_corr_thres=0.03, minimum_lag=5e-3, clean_df=False,
                         verbose=True):
    """Remove bad fit and pick the best of remaining

    Inner function of sync_by_correlation

    frames_df has bef, center and aft crosscorrelation, find which one are reasonable
    and pick the best among those

    Args:
        frames_df (pd.DateFrame): Dataframe containing crosscorrelation information
        frame_log (pd.DateFrame): Dataframe from bonsai logger
        correlation_threshold (float): threshold on the pearson correlation. Anything
                                       below is considered a failure to fit
        relative_corr_thres (float): When multiple correlations are above threshold,
                                     will consider only among those that less than
                                     `relative_corr_thres` away from max corr
        minimum_lag (float): Minimum possible lag. Anything below is considered a
                             failure to fit
        clean_df (bool): Should the output contain all columns) (if clean_df=False,
                         default) or only the one selected and not the `bef`, `center`
                         and `aft` version
        verbose (bool): Print progress?

    Returns:
        df (pd.DataFrame): A dataframe with one of 3 crosscorr selected.
    """
    good = np.logical_and(
        frames_df['closest_frame_center'] == frames_df['closest_frame_bef'],
        frames_df['closest_frame_bef'] == frames_df['closest_frame_aft'])
    frames_df['closest_frame'] = np.nan
    frames_df['lag'] = np.nan
    frames_df['sync_reason'] = 'not done'
    frames_df['crosscorr_picked'] = 'not done'

    # for these I can take whichever
    frames_df.loc[good, 'closest_frame'] = frames_df.loc[good, 'closest_frame_bef']
    frames_df.loc[good, 'crosscorr_picked'] = 'bef'
    frames_df.loc[good, 'lag'] = frames_df.loc[good, 'lag_bef']
    frames_df.loc[good, 'sync_reason'] = 'consensus'
    if verbose:
        print("Sync'ed %d frames easily. That's %d%% of the recording." % (
            np.sum(good), np.sum(good) / len(good) * 100), flush=True)
    labels = ['bef', 'center', 'aft']
    # first let's get rid of very bad fit
    did_not_fit = frames_df.loc[~good, ['peak_corr_%s' % l for l in labels]].values < \
                  correlation_threshold
    # and impossible lags
    impossible_lag = frames_df.loc[~good, ['lag_%s' % l for l in labels]] < minimum_lag
    bad = did_not_fit | impossible_lag
    if verbose:
        start = time.time()
        print('Matching frame to logger', flush=True)

    for ind, line in bad.iterrows():
        n_bad = np.sum(line)
        if n_bad == 3:
            # some frames are not fitted at all
            frames_df.loc[ind, 'sync_reason'] = 'not synced'
            frames_df.loc[ind, 'crosscorr_picked'] = 'none'
            continue
        elif n_bad == 2:
            # some there is only one fit remaining, keep this one
            lab = labels[np.where(~line.values)[0][0]]
            frames_df.loc[ind, 'lag'] = frames_df.loc[ind, 'lag_%s' % lab]
            frames_df.loc[ind, 'closest_frame'] = frames_df.loc[
                ind, 'closest_frame_%s' % lab]
            frames_df.loc[ind, 'sync_reason'] = 'only fit'
            frames_df.loc[ind, 'crosscorr_picked'] = lab
            continue
        elif n_bad == 1:
            # one was bad, if the other 2 agree, pick that
            lab = list(labels)
            lab.pop(np.where(line.values)[0][0])
            if frames_df.loc[ind, 'closest_frame_%s' % lab[0]] == frames_df.loc[
                ind, 'closest_frame_%s' % lab[1]]:
                frames_df.loc[ind, 'lag'] = frames_df.loc[ind, 'lag_%s' % lab[0]]
                frames_df.loc[ind, 'closest_frame'] = frames_df.loc[
                    ind, 'closest_frame_%s' % lab[0]]
                frames_df.loc[ind, 'sync_reason'] = 'partial consensus'
                frames_df.loc[ind, 'crosscorr_picked'] = lab[0]
                continue
        else:
            lab = list(labels)
        # We have 2 or 3 reasonable values.
        # get rid of the less good correlation
        corrs = np.array(frames_df.loc[ind, ['peak_corr_%s' % l for l in lab]],
                         dtype=float)
        corrs[np.isnan(corrs)] = 0
        good = np.nanmax(corrs) - corrs < relative_corr_thres
        reason = ['relative peak corr'] if any(~good) else []
        lab = [l for ok, l in zip(good, lab) if ok]
        if len(lab) > 1:
            reason.append('photodiode matching')
        # Among remaining, pick that which is the closest to photodiode
        val = frames_df.loc[ind, ['quadcolor_%s' % l for l in lab]]
        closest = np.abs(val - frames_df.loc[ind, 'photodiode']).values.argmin()
        lab = lab[closest]
        frames_df.loc[ind, 'lag'] = frames_df.loc[ind, 'lag_%s' % lab]
        frames_df.loc[ind, 'closest_frame'] = frames_df.loc[ind, 'closest_frame_%s' % lab]
        frames_df.loc[ind, 'sync_reason'] = ' & '.join(reason)
        frames_df.loc[ind, 'crosscorr_picked'] = lab

    if verbose:
        end = time.time()
        print('done (%d s)' % (end - start), flush=True)

    # after the initial match, clean-up parts where the order is wrong
    time_travel = np.where(np.diff(frames_df.closest_frame.values) < 1)[0]
    time_travel = time_travel[(time_travel > 0) & (time_travel < len(frames_df) - 2)]
    # Two options: either frame n is shifted forward or n+1 is shifted backward,
    # just look at both
    for shift in [0, 1]:
        baddies = time_travel + shift
        nm1 = frames_df.closest_frame[baddies - 1].values
        np1 = frames_df.closest_frame[baddies + 1].values
        n = frames_df.closest_frame[baddies].values
        to_replace = baddies[(np1 - nm1) == 2]
        # set lag to unknown
        frames_df.loc[to_replace, 'lag'] = np.nan
        reason = 'fixing time travel (%s)' % ('to future' if not shift else 'to past')
        frames_df.loc[to_replace, 'sync_reason'] = reason
        frames_df.loc[to_replace,
                      'closest_frame'] = frames_df.loc[to_replace - 1,
                                                       'closest_frame'].values + 1

    # Finally add the color from the sequence
    frames_df['quadcolor'] = np.nan
    matched = ~np.isnan(frames_df.closest_frame)
    frames_df.loc[matched, 'quadcolor'] = frame_log.loc[frames_df.loc[matched,
                                                                      'closest_frame'],
                                                        'PhotoQuadColor'].values
    if clean_df:
        cols = [c for c in frames_df.columns if (not c.endswith('bef')) and
                (not c.endswith('center')) and (not c.endswith('aft'))]
        return pd.DataFrame(frames_df[cols])
    return frames_df
