import time
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.io_module import harp
import pandas as pd
import flexiznam as flm

from cottage_analysis.utilities.time_series_analysis import searchclosest
from cottage_analysis.utilities import continuous_data_analysis as cda


PROJECT = 'hey2_3d-vision_foodres_20220101'
MOUSE = 'Test'
SESSION = 'S20221013'
RECORDING = 'R203221_SpheresPermTubeReward'
MESSAGES = 'Test_S20221013_R203221_SpheresPermTubeReward_harpmessage.bin'


def test_detect_frame_sequence():
    # load data
    data_root = flm.PARAMETERS['data_root']
    msg = Path(data_root['raw']) / PROJECT / MOUSE / SESSION / RECORDING / MESSAGES
    p_msg = Path(data_root['processed']) / PROJECT / MOUSE / SESSION / RECORDING
    p_msg = p_msg / (msg.stem + '.npz')
    if p_msg.is_file():
        harp_messages = np.load(p_msg)
    else:
        harp_messages = harp.load_harp(msg)
        p_msg.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p_msg, **harp_messages)

    frame_log = pd.read_csv(msg.parent / 'FrameLog.csv')
    expected_sequence = pd.read_csv(msg.parent / 'random_sequence.csv',
                                    header=None).loc[:, 0].values
    step_values = frame_log.PhotoQuadColor.unique()
    ao_time = harp_messages['analog_time']
    photodiode = harp_messages['photodiode']
    ao_sampling = 1 / np.mean(np.diff(ao_time))
    # data loaded

    # run all
    frame_rate = 144
    frames_df, extra_out = find_frames.sync_by_correlation(frame_log, ao_time, photodiode,
                                                time_column='HarpTime',
                                                sequence_column='PhotoQuadColor',
                                                num_frame_to_corr=6,
                                                maxlag=3. / frame_rate,
                                                expected_lag=2. / frame_rate,
                                                frame_rate=frame_rate,
                                                correlation_threshold=0.8,
                                                minimum_lag=1. / frame_rate,
                                                do_plot=True, verbose=True, debug=True)

    db = extra_out['debug_info']
    normed_pd = np.array(photodiode, dtype=float)
    normed_pd -= np.quantile(normed_pd, 0.01)
    normed_pd /= np.quantile(normed_pd, 0.99)

    rng = np.random.default_rng(102)
    w = frames_df[frames_df.sync_reason == 'photodiode matching'].index
    random_select = [w[i] for i in rng.integers(len(w), size=10)]
    bad = np.diff(frames_df.closest_frame.values) < 1
    badi = np.where(bad)[0]
    random_select = frames_df.iloc[badi[100] + np.array([0, 1], dtype=int)].index
    labels = ['bef', 'center', 'aft']
    num_frame_to_corr = 5
    maxlag = int(5. / frame_rate * ao_sampling)
    expected_lag = int(2. / frame_rate * ao_sampling)
    window = [np.array([-1, 1]) * maxlag +
              np.array(w * num_frame_to_corr / frame_rate * ao_sampling, dtype='int')
              for w in [np.array([-1, 0]), np.array([-0.5, 0.5]), np.array([0, 1])]]
    seq_trace = db['seq_trace']

    for frame in random_select:
        # frame = frames_df[~good].index[num]
        # frame = frames_df.index[num]
        fseries = frames_df.loc[frame]
        on_s = fseries.onset_sample
        off_s = fseries.offset_sample
        on_t = fseries.onset_time
        off_t = fseries.offset_time
        w = np.array([-50, 50])
        vfdf = frames_df[(frames_df.onset_sample > w[0] + on_s) &
                         (frames_df.offset_sample < w[1] + off_s)]
        qc = np.array([fseries[['quadcolor_%s' % w for w in labels]]])
        best = fseries.crosscorr_picked
        fig = plt.figure(figsize=(7, 7))
        plt.gca().get_yaxis().set_visible(False)

        col = dict(bef='r', center='g', aft='b')
        for i in range(3):
            label = 'Photodiode' if i == 1 else None
            plt.plot(ao_time[slice(*w + on_s)] - on_t, normed_pd[slice(*w + on_s)] + i,
                     label=label, color='purple')
            label = 'Frame #%d' % frame if i == 1 else None
            plt.axvspan(0, off_t - on_t, color='purple', alpha=0.2, label=label)
            plt.plot(fseries.peak_time - on_t, fseries.photodiode, 'o', color='purple')

        vlog = frame_log[
            (frame_log.HarpTime > w[0] / ao_sampling + on_t - fseries.lag_bef) &
            (frame_log.HarpTime < w[1] / ao_sampling + off_t)]
        plt.plot(vlog.HarpTime.values - on_t, vlog.PhotoQuadColor - 1.5,
                 drawstyle='steps-post',
                 label='Render frame')

        i = 0
        for win, lab in zip(window, ['bef', 'center', 'aft']):
            cut_win = win + maxlag * np.array([1, -1], dtype=int)
            l = fseries['lag_%s' % lab]
            part = seq_trace[slice(*win + on_s)]
            cut_part = seq_trace[slice(*cut_win + on_s)]
            x = normed_pd[slice(*win + on_s)][maxlag:-maxlag + 1]

            plt.plot(ao_time[slice(*win + on_s)] - on_t + l, part + i, alpha=0.75, lw=2,
                     color=col[lab])
            plt.plot(ao_time[slice(*win + on_s)][maxlag:-maxlag + 1] - on_t,
                     x + i, alpha=0.5, lw=4, ls='--', color=col[lab])

            cl = fseries['closest_frame_%s' % lab]
            plt.plot(frame_log.iloc[cl].HarpTime - on_t,
                     frame_log.iloc[cl].PhotoQuadColor - 1.5 + i / 6, 'o',
                     color=col[lab])
            if lab == best:
                plt.plot(frame_log.iloc[cl].HarpTime - on_t,
                         frame_log.iloc[cl].PhotoQuadColor - 1.5 + i / 6, 'o',
                         mfc='None', mec='k', ms=10, mew=2)
                plt.plot(frame_log.iloc[cl].HarpTime - on_t + l,
                         frame_log.iloc[cl].PhotoQuadColor + i, 'o',
                         color='k')
                plt.plot(ao_time[slice(*cut_win + on_s)] - on_t + l, cut_part + i,
                         alpha=1, lw=1,
                         color='k', label='Selected match')

            i += 1
            plt.title('%s' % fseries.onset_sample)

        plt.legend(loc='lower right')

    ####################################
    ###OLD stuff
    raise IOError()
    plt.figure()
    plt.axvspan(0, off_t-on_t, color='purple', alpha=0.2)
    plt.plot(ao_time[slice(*w+on_s)]-on_t, normed_pd[slice(*w+on_s)] + 2,
             label='Photodiode')
    plt.scatter(vfdf.peak_time - on_t, vfdf.photodiode + 2, marker='.', color='k')
    plt.plot(fseries.peak_time - on_t, fseries.photodiode + 2, 'o', color='k')
    plt.plot(vlog.HarpTime.values - on_t, vlog.PhotoQuadColor,
             drawstyle='steps-post', label='Render frame')
    plt.plot(vlog.HarpTime.values - on_t + fseries.lag_bef, vlog.PhotoQuadColor + 1,
             drawstyle='steps-post', label='Lagged render frame')
    cl_f = frame_log.iloc[fseries.closest_frame_bef]
    plt.plot(cl_f.HarpTime - on_t, cl_f.PhotoQuadColor, 'o', color='k')
    plt.plot(cl_f.HarpTime - on_t + fseries.lag_bef, cl_f.PhotoQuadColor + 1, 'o',
             color='k')
    plt.plot(fseries.peak_time - on_t, fseries.photodiode + 2, 'o', color='k')
    plt.show()

    frames_df[(frames_df.quadcolor == 1) & (frames_df.photodiode < 0.5)]
    plt.scatter(frames_df.quadcolor, frames_df.photodiode)
    plt.show()
    # first do batch of consecutive frames
    skips = np.where(frames_df.include_skip | frames_df.is_jump)[0]
    n_contiguous = np.diff(skips)
    valid_batch = n_contiguous > num_frame_to_corr * 2
    for ibatch in np.where(valid_batch)[0]:
        # in a continuous batch, the frame after num_frame_to_corr, should be nicely
        # synced
        b, e = skips[ibatch:ibatch+2]
        full_sync = [b + num_frame_to_corr, e - num_frame_to_corr]
        lags = frames_df.iloc[slice(*full_sync)].lag
        assert np.max(np.abs(lags-lags.iloc[0])) < 1/frame_rate
        plt.plot(frames_df.iloc[b:e].lag)
        plt.plot(lags)
        plt.show()

        plt.plot(ao_time[frames_df.onset_sample[b]:frames_df.offset_sample[e]],
                 normed_pd[frames_df.onset_sample[b]:frames_df.offset_sample[e]])
        plt.plot(ao_time[frames_df.onset_sample[
                             b+num_frame_to_corr]:frames_df.offset_sample[e-num_frame_to_corr]],
                 normed_pd[frames_df.onset_sample[b+num_frame_to_corr]:frames_df.offset_sample[
                     e-num_frame_to_corr]])
        plt.plot(frames_df.onset_time[b:e], frames_df.photodiode[b:e],
                 drawstyle='steps-post')
        plt.plot(frames_df.onset_time[b:e], frames_df.is_jump[b:e],
                 drawstyle='steps-post')
        plt.show()
        if any(np.abs(lags-lags.iloc[0]) > 4e-3):
            continue


    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    img = ax.imshow(cc_mat, aspect='auto', origin='lower',
                    extent=(lags_time[0] * 1e3, lags_time[-1] * 1e3, 0, len(frames_df)))
    cb = plt.colorbar(img, ax=ax)
    cb.set_label('Pearson correlation')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Frame #')

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(delay_time * 1000, peak_corr, alpha=0.2, c=frames_df.include_skip)
    ax.set_xlabel('Frame lag (ms)')
    ax.set_ylabel('Peak correlation')

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 2)
    # check a random frame
    i_frame = 20
    fseries = frames_df.iloc[i_frame]
    cl_series = frame_log.iloc[fseries.closest_frame]
    o = fseries.onset
    w = np.array([-100, 100])
    ax.plot(ao_time[slice(*w+o)], normed_pd[slice(*w+o)])
    ax.axvspan(ao_time[o], ao_time[fseries.offset], color='purple', alpha=0.2)
    ax.axvline(cl_series.HarpTime, color='Grey', alpha=0.2, ls='--')
    ax.axvline(cl_series.HarpTime + fseries.lag, color='Grey', alpha=0.2, ls='-')
    ax.plot(frame_log.iloc[[fseries.closest_frame, fseries.closest_frame +
                           1]].HarpTime.values + fseries.lag,
            [fseries.quadcolor]*2)
    for i in np.arange(-10, 10):
        ser = frames_df.iloc[i + i_frame]
        ax.plot(ao_time[[ser.onset, ser.offset]], [ser.quadcolor] * 2)
    v = ((ao_time[frames_df.onset] > ao_time[o + w[0]]) &
         (ao_time[frames_df.onset] < ao_time[o + w[1]]))
    plt.plot(ao_time[frames_df.onset[v]], (frames_df.lag[v] - fseries.lag)*100)
    plt.ylim([-0.1, 1.1])
    plt.show()
    v = (frame_log.HarpTime.values > ao_time[o + w[0]]) & (frame_log.HarpTime.values <
                                                           ao_time[o + w[1]])

    plt.plot(frame_log.HarpTime.values[v] + fseries.lag,
             frame_log.PhotoQuadColor.values[v])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)





    frame_pk_i = searchclosest(ao_time[frame_indices],
                               frame_log.HarpTime.values + delay_time)
    pks = normed_pd[frame_indices][frame_pk_i]
    frame_seq = frame_log.PhotoQuadColor.values
    df = pd.DataFrame(dict(p0=pks[:-1], p1=pks[1:], s0=frame_seq[:-1], s1=frame_seq[1:]))
    avg = df.groupby(['s0', 's1']).aggregate(np.nanmean).reset_index()
    avg['x'] = np.array(avg['s0'] * 4, dtype=int)
    avg['y'] = np.array(avg['s1'] * 4, dtype=int)
    amp_mat = np.zeros((5, 5)) + np.nan
    expected = np.zeros((5, 5))
    for i in range(5):
        expected[i, :] = i * 0.25
    for x, y, v in avg[['x', 'y', 'p1']].values:
        amp_mat[int(x), int(y)] = v
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = ax.imshow(amp_mat.T, origin='lower', cmap='RdBu_r')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(np.arange(0, 1.2, 0.25))
    ax.set_yticklabels(np.arange(0, 1.2, 0.25))
    ax.set_ylabel('Previous colour')
    ax.set_xlabel('Current colour')
    cb = plt.colorbar(img, ax=ax)
    cb.set_label('Error (average)')
    plt.show()

    corrected_seq = sequence
    seqind = np.array(sequence * 4, dtype=int)
    correction_mat = amp_mat - expected
    np.fill_diagonal(correction_mat, 0)
    correction = correction_mat[seqind[1:], seqind[:-1]]
    corrected_seq[1:] += correction
    corr_seq_trace, corr_ideal_pd = find_frames.ideal_photodiode(time_base=ao_time,
                                                       switch_time=switch_time,
                                                       sequence=corrected_seq)
    start = time.time()
    print('Starting', flush=True)
    cc_mat_corr = np.zeros((len(frame_onset_index), maxlag * 2))
    for iframe, foi in enumerate(frame_onset_index):
        corr, lags = cda.crosscorrelation(normed_pd[slice(*window + foi)],
                                          corr_ideal_pd[slice(*window + foi)],
                                          maxlag=maxlag,
                                          expected_lag=expected_lag,
                                          normalisation='pearson')
        cc_mat_corr[iframe] = corr
    end = time.time()
    print('done (%d s)' % (end - start))
    plt.subplot(2,2,1)
    plt.imshow(cc_mat, aspect='auto')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(cc_mat_corr, aspect='auto')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(cc_mat-cc_mat_corr, aspect='auto', cmap='RdBu_r')
    plt.colorbar()
    plt.show()
    frame_dfs = []
    for f_index, frame_df in frame_log.iterrows():
        kwargs = dict(photodiode_time=ao_time, photodiode_signal=normed_pd,
                      switch_time=frame_log.HarpTime.values,
                      sequence=frame_log.PhotoQuadColor.values,
                      num_frame_to_corr=num_frame_to_corr,
                      maxlag=int(5 / frame_rate * ao_sampling),
                      expected_lag=int(3 / frame_rate * ao_sampling), return_chunk=True)
        delay, corr, chk = find_frames.sync_by_correlation(frame_df.HarpTime, **kwargs)
        frame_df['DisplayTime'] = frame_df.HarpTime + delay / ao_sampling
        frame_df['Delay'] = delay
        frame_df['Correlation'] = corr
        frame_dfs.append(frame_df)
    end = time.time()
    print('done (%d s)' % (end - start))
    frame_dfs = pd.DataFrame(frame_dfs)

    def plot_one_frame(frame_series, window=np.array([-0.1, 0.1])):
        frame_time = frame_series.HarpTime
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axvline(frame_time, color='Grey', alpha=0.5, ls='--')
        ax.axvline(frame_series.DisplayTime, color='Grey', alpha=0.5, ls='-')
        ax.axvline(frame_series.DisplayTimeOffset, color='Grey', alpha=0.5, ls='-')
        b, e = ao_time.searchsorted(window + frame_time)
        ax.plot(ao_time[b:e], normed_pd[b:e])
        seq_trace, ideal_pd = find_frames.ideal_photodiode(time_base=ao_time[b:e],
                                                           switch_time=frame_log.HarpTime.values,
                                                           sequence=frame_log.PhotoQuadColor.values)
        ax.plot(ao_time[b:e], seq_trace, drawstyle='steps-post')
        ax.plot(ao_time[b:e] + frame_series.Delay / ao_sampling, ideal_pd)
        ax.set_xlim(frame_time + np.array([-50e-3, 50e-3]))
        plt.show()

    frame_df = frame_dfs[frame_dfs.Correlation > 12].iloc[102]
    frame_dfs['DisplayTimeOffset'] = np.nan
    frame_dfs['DisplayTimeOffset'].iloc[:-1] = frame_dfs['DisplayTime'].iloc[1:]
    plot_one_frame(frame_df)
    plt.scatter(frame_dfs.Delay, frame_dfs.Correlation, alpha=0.1)
    plt.show()
    if False:
        # plot
        b = frames_in_bunch[0] - 7 * 5
        e = frames_in_bunch[-1] + 7 * 5
        plt.subplot(2, 1, 1)
        plt.plot(ao_time[b:e], normed_pd[b:e])
        plt.plot(chunk_time, normed_pd[chunk])
        plt.plot(chunk_time + shift_time, ideal_pd)
        ok = (frame_indices < e) & (frame_indices > b)
        plt.plot(ao_time[frame_indices[ok]], normed_pd[frame_indices[ok]], '.')
        plt.plot(ao_time[frames_in_bunch], normed_pd[frames_in_bunch], 'o')
        plt.plot(seq.HarpTime.values + shift_time,
                 seq.PhotoQuadColor,  drawstyle='steps-post')
        plt.axvline(frame_time)
        plt.subplot(2, 1, 2)
        plt.plot(lags/ao_sampling*1000, corr)
        plt.show()

    t0 = harp_messages['analog_time'][0]
    ao_time = harp_messages['analog_time'] - t0
    photo = np.array(harp_messages['photodiode'], dtype=float)
    photo -= photo.min()
    photo /= photo.max()
    b = frame_indices[1:][np.diff(frame_indices).argmin()] / ao_sampling - 0.05
    e = b + 0.1
    b_i, e_i = ao_time.searchsorted([b, e])
    local_t0 = ao_time[b_i]
    valid_f = np.logical_and(frame_indices > b_i, frame_indices < e_i)
    plt.plot(ao_time[frame_indices[valid_f]]-local_t0, photo[frame_indices[valid_f]], 'o')
    plt.plot(ao_time[b_i: e_i]-local_t0, photo[b_i:e_i])
    b_i, e_i = pd_log.HarpTime.searchsorted([b + t0, e + t0])
    lag = 20e-3
    plt.plot(pd_log.HarpTime.values[b_i:e_i] - t0 - local_t0 + lag,
             pd_log.PhotoQuadColor.values[b_i:e_i], zorder=-10)
    plt.show()