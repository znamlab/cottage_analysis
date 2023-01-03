# !!! NEED TO CHANGE THESE:
# 1) add font choices for title, axis_ticks, axis_label for each plotting function

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from typing import Sequence, Dict, Any
import scipy


def segment_arr(arr_idx, segment_size):
    '''
    Segment an input array into small chunks with defined size.

    :param arr_idx: np.ndarray. 1D. Array of indices of a target array.
    :param segment_size: int. Number of elements desired for each segment.
    :return: segment_starts: list. List of starting indices for all segments.
             segment_ends: list. List of end indices for all segments.
    '''
    batch_num = len(arr_idx)//segment_size
    segment_starts = np.arange(0,batch_num*segment_size+1,segment_size)
    segment_ends = np.arange(segment_size,batch_num*segment_size+segment_size,segment_size)
    if len(arr_idx)%segment_size!=0:
        segment_ends = np.concatenate((segment_ends, (arr_idx[-1]+1).reshape(-1)))
    segment_starts = (segment_starts+arr_idx[0])[:len(segment_ends)]
    segment_ends = segment_ends+arr_idx[0]
    return segment_starts, segment_ends


def calculate_R_squared(actual_data, predicted_data):
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    residual_var = np.sum((predicted_data-actual_data)**2)
    total_var = np.sum((actual_data - np.mean(actual_data))**2)
    R_squared = 1-residual_var/total_var
    return R_squared


def get_confidence_interval(arr, sem=[], sig_level=0.05, mean_arr=[]):
    '''
    Get confidence interval of an input array.

    :param arr: np.ndarray. For example, ntrials x time to calculate confidence interval across trials.
    :param sem: np.ndarray. For example, 1 x time. Default nan. If you have calculated SEM outside, you can provide SEM.
    :param sig_level: float. Significant level. Default 0.05.
    :param mean_arr:
    :return:
    '''
#     CI_low, CI_high = scipy.stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
#     CI_low, CI_high = scipy.stats.norm.interval(0.95, loc=np.mean(arr), scale=st.sem(arr))
    z = scipy.stats.norm.ppf((1-sig_level/2))
    if len(sem) > 0:
        sem = sem
    else:
        sem = scipy.stats.sem(arr,nan_policy='omit')
    if len(mean_arr)>0:
        CI_low = mean_arr - z*sem
        CI_high = mean_arr + z*sem
    else:
        CI_low = np.average(arr, axis=0) - z*sem
        CI_high = np.average(arr, axis=0) + z*sem
    return CI_low, CI_high


def plot_raster(arr, vmin, vmax, cmap, blank_period=5, frame_rate=30, title=None, suffix=None, title_on=False, fontsize=10, extent=[], set_nan_cmap=True):
    '''
    Raster plot of input params. Row: trials. Column: time.

    :param arr: np.ndarray. 2D array, trials x time.
    :param vmin: vmin for plt.imshow heatmap.
    :param vmax: vmax for plt.imshow heatmap.
    :param cmap: color map object.
    :param blank_period: float, blank period in seconds.
    :param frame_rate: int, frame rate for imaging.
    :param title: str, title of the plots.
    :param suffix: str, suffix added to the end of the title.
    :param title_on: bool, whether put title + suffix on or only put the suffix.
    :param fontsize: int, fontsize of title.
    :param extent: list, [extent_x_min, extent_x_max, extent_y_min, extent_y_max]. Min/max values for heatmap x and y axes.
    :param set_nan_cmap: bool, whether to set nan values in colormap as a different colour (silver)
    '''
    current_cmap = mpl.cm.get_cmap(cmap).copy()
    if set_nan_cmap:
        current_cmap.set_bad(color='silver')
    if len(extent) > 0:
        extent = extent
    else:
        extent = [-blank_period, -blank_period + arr.shape[1] / frame_rate, arr.shape[0], 1]
    plt.imshow(arr, aspect='auto', vmin=vmin, vmax=vmax, cmap=current_cmap, extent=extent, rasterized=False,
               interpolation='none')
    plt.colorbar()
    if title_on:
        plt.title(title + ' ' + suffix, fontsize=fontsize)
    else:
        plt.title(suffix, fontsize=fontsize)
    #     plt.xticks(np.arange(0,arr.shape[1],500))
    plt.ylim([arr.shape[0], 1])


def plot_trial_onset_offset(onset, offset, ymin, ymax):
    '''
    Plot a dashed line for trial onset and offset (based on x axis) on an existing plot.

    :param onset: float, onset time in whatever unit x axis of existing plot is.
    :param offset: float, onset time in whatever unit y axis of existing plot is.
    :param ymin: float, min value where dashed line starts at y axis.
    :param ymax: float, max value where dashed line starts at y axis.
    :return: None.
    '''
    plt.vlines(x=onset, ymin=ymin, ymax=ymax, linestyles='dashed', colors='k', alpha=0.5, linewidth=1, rasterized=False)
    plt.vlines(x=offset, ymin=ymin, ymax=ymax, linestyles='dashed', colors='k', alpha=0.5, linewidth=1 ,rasterized=False)


def get_confidence_interval(arr, sem=[], sig_level=0.05, mean_arr=[]):
    '''
    Calculate the confidence interval for an input array (shape: n_trials x n_samples)

    :param arr: np.ndarray (n_trials x n_samples), array to compute confidence interval,
    :param sem: np.ndarray (1D array), default []. Input SEM of input array if already calculated.
    :param sig_level: float, default 0.05.
    :param mean_arr: np.ndarray (1D array), default []. Input mean of input array if already calculated.
    :return: CI_low: np.ndarray (1D array), lower boundary of confidence interval.
             CI_high: np.ndarray (1D array), higher boundary of confidence interval.
    '''
#     CI_low, CI_high = scipy.stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
#     CI_low, CI_high = scipy.stats.norm.interval(0.95, loc=np.mean(arr), scale=st.sem(arr))
    z = scipy.stats.norm.ppf((1-sig_level/2))
    if len(sem) > 0:
        sem = sem
    else:
        sem = scipy.stats.sem(arr,nan_policy='omit')
    if len(mean_arr)>0:
        CI_low = mean_arr - z*sem
        CI_high = mean_arr + z*sem
    else:
        CI_low = np.average(arr, axis=0) - z*sem
        CI_high = np.average(arr, axis=0) + z*sem
    return CI_low, CI_high


def plot_line_with_error(arr, CI_low, CI_high, linecolor, label=None, marker='-', markersize=None, xarr=[], xlabel=None, ylabel=None, title_on=False, title=None, suffix=None, fontsize=10, axis_fontsize=15, linewidth=0.5):
    if len(xarr) == 0:
        plt.plot(arr, marker, c = linecolor, linewidth=linewidth, label=label, alpha = 1, markersize=markersize, rasterized=True)
        plt.fill_between(np.arange(len(arr)), CI_low, CI_high, color=linecolor, alpha=0.3, edgecolor=None, rasterized=True)
    else:
        plt.plot(xarr, arr, marker, c = linecolor, linewidth=linewidth, label=label, alpha = 1, markersize=markersize, rasterized=True)
        plt.fill_between(xarr, CI_low, CI_high, color=linecolor, alpha=0.3, edgecolor=None, rasterized=True)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    if title_on:
        plt.title(title+' '+suffix, fontsize=fontsize)
    else:
        plt.title(suffix, fontsize=fontsize)


def plot_scatter(x,y, xlim, ylim, markercolor, xlabel=None, ylabel=None, title_on=False, title=None, label=None):
    plt.plot(x, y, 'o', markersize=3, c=markercolor, rasterized=True, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if title_on:
        plt.title(title)


def get_binned_stats(xarr, yarr, bin_number):
    binned_stats = {
        'bin_edge_min': [],
        'bin_edge_max': [],
        'bins': [],
        'bin_means': np.zeros((xarr.shape[0], bin_number)),
        #         'bin_values_all':[[[None for i in range(bin_number)] for j in range(trace_arr_noblank.shape[1])] for q in range(trace_arr_noblank.shape[0])],
        'bin_stds': np.zeros((xarr.shape[0], bin_number)),
        'bin_counts': np.zeros((xarr.shape[0], bin_number)),
        'bin_centers': np.zeros((xarr.shape[0], bin_number)),
        'bin_edges': np.zeros((xarr.shape[0], bin_number + 1)),
        'binnumber': []
    }
    binned_stats['bin_edge_min'] = np.round(np.nanmin(xarr), 1)
    binned_stats['bin_edge_max'] = np.round(np.nanmax(xarr), 1) + 0.1
    binned_stats['bins'] = np.linspace(start=binned_stats['bin_edge_min'], stop=binned_stats['bin_edge_max'],
                                       num=bin_number + 1, endpoint=True)

    for i in range(xarr.shape[0]):
        bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(xarr[i].flatten(), yarr[i].flatten(),
                                                                       statistic='mean', bins=binned_stats['bins'])
        bin_stds, _, _ = scipy.stats.binned_statistic(xarr[i].flatten(), yarr[i].flatten(), statistic='std',
                                                      bins=binned_stats['bins'])
        bin_counts, _, _ = scipy.stats.binned_statistic(xarr[i].flatten(), yarr[i].flatten(), statistic='count',
                                                        bins=binned_stats['bins'])
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        binned_stats['bin_means'][i, :] = bin_means
        binned_stats['bin_stds'][i, :] = bin_stds
        binned_stats['bin_counts'][i, :] = bin_counts
        binned_stats['bin_centers'][i, :] = bin_centers
        binned_stats['bin_edges'][i, :] = bin_edges
        binned_stats['binnumber'].append(binnumber.reshape(xarr.shape[1], -1))

    #     for i in range(xarr.shape[0]):
    #         for j in range(xarr.shape[1]):
    #             for b in range(bin_number):
    #                 binned_stats['bin_values_all'][i][j][b] = yarr[i][j][np.where(binned_stats['binnumber'][i][j]==b+1)[0]]

    return binned_stats


def get_binned_arr(xarr, yarr, bin_number, bin_edge_min=0, bin_edge_max=6):
    '''
    assume xarr and yarr are 3D arrays
    '''
    binned_stats = {}
    binned_yrr = np.zeros((yarr.shape[0], yarr.shape[1], bin_number))
    bins = np.linspace(start=bin_edge_min, stop=bin_edge_max, num=bin_number + 1, endpoint=True)

    for i in range(xarr.shape[0]):
        for j in range(xarr.shape[1]):
            bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x=xarr[i, j, :][~np.isnan(xarr[i, j, :])],
                                                                           values=yarr[i, j, :][
                                                                               ~np.isnan(yarr[i, j, :])],
                                                                           statistic='mean', bins=bins)
            binned_yrr[i, j, :] = bin_means
    binned_stats['bins'] = bins
    binned_stats['binned_yrr'] = binned_yrr

    return binned_stats


def plot_dFF_binned_speed(speed_arr, trace_arr, speed_bins, depth_list,
                          plot_rows, plot_cols, plot_row, plot_col, title, xlabel, linecolors,
                          log=False,
                          xlim=[], ylim=[], speed_arr_blank=[], trace_arr_blank=[], fontsize=20, axis_fontsize=15):
    binned_stats = get_binned_stats(xarr=speed_arr, yarr=trace_arr, bin_number=speed_bins)
    CI_lows = np.zeros(((len(depth_list)), speed_bins))
    CI_highs = np.zeros(((len(depth_list)), speed_bins))
    for idepth in range(len(depth_list)):
        with np.errstate(divide='ignore', invalid='ignore'):
            CI_lows[idepth], CI_highs[idepth] = get_confidence_interval(arr=[],
                                                                        sem=np.divide(binned_stats['bin_stds'][idepth],
                                                                                      np.sqrt(
                                                                                          binned_stats['bin_counts'][
                                                                                              idepth])),
                                                                        sig_level=0.05,
                                                                        mean_arr=binned_stats['bin_means'][idepth])

        #         for idepth, linecolor in zip(range(len(depth_list)),['skyblue','skyblue','skyblue']):
        #             plt.subplot2grid([plot_rows,plot_cols],[idepth,plot_col])
        #             CI_low = CI_lows[idepth]
        #             CI_high = CI_highs[idepth]
        #             if idepth == 0:
        #                 title_on = True
        #             else:
        #                 title_on = False
        #             plot_line_with_error(xarr=binned_stats['bin_centers'][idepth,:],
        #                                  arr=binned_stats['bin_means'][idepth,:],
        #                                  CI_low=CI_low, CI_high=CI_high,
        #                                  linecolor=linecolor, label=None, marker='o-', markersize=3,
        #                                  xlabel=xlabel, ylabel='dFF',
        #                                  title_on=title_on, title=title,
        #                                  suffix='Depth='+str(int(depth_list[idepth]*100))+'cm', fontsize=fontsize)
        #             plt.ylim([np.nanmin(CI_lows),np.nanmax(CI_highs)])
        #             plt.xlim([binned_stats['bin_edge_min'],binned_stats['bin_edge_max']])

        plt.subplot2grid([plot_rows, plot_cols], [plot_row, plot_col])
        if log == False:
            for idepth, linecolor in zip(range(len(depth_list)), linecolors):
                if idepth == 0:
                    title_on = True
                else:
                    title_on = False
                plot_line_with_error(xarr=binned_stats['bin_centers'][idepth, :],
                                     arr=binned_stats['bin_means'][idepth, :],
                                     CI_low=CI_lows[idepth], CI_high=CI_highs[idepth],
                                     linecolor=linecolor, label=str(int(depth_list[idepth] * 100)) + 'cm', marker='o-',
                                     markersize=3,
                                     xlabel=xlabel, ylabel='dF/F',
                                     title_on=title_on, title='', fontsize=fontsize, suffix='')
            if (len(trace_arr_blank) > 0) and (len(speed_arr_blank) > 0):
                binned_stats_blank = get_binned_stats(xarr=speed_arr_blank, yarr=trace_arr_blank, bin_number=speed_bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    CI_low, CI_high = get_confidence_interval(arr=[],
                                                              sem=np.divide(binned_stats_blank['bin_stds'][0], np.sqrt(
                                                                  binned_stats_blank['bin_counts'][0])),
                                                              sig_level=0.05,
                                                              mean_arr=binned_stats_blank['bin_means'][0])

                plot_line_with_error(xarr=binned_stats_blank['bin_centers'][0, :],
                                     arr=binned_stats_blank['bin_means'][0, :],
                                     CI_low=CI_low, CI_high=CI_high,
                                     linecolor='gray', label='blank', marker='o-', markersize=3,
                                     xlabel=xlabel, ylabel='dF/F',
                                     title_on=False, title='', fontsize=fontsize, axis_fontsize=axis_fontsize,
                                     suffix='')
                plt.ylim([np.nanmin(np.concatenate((CI_lows.flatten(), CI_low.flatten()))),
                          1.2 * np.nanmax(np.concatenate((CI_highs.flatten(), CI_high.flatten())))])
                plt.xlim([np.nanmin([binned_stats['bin_edge_min'], binned_stats_blank['bin_edge_min']]),
                          np.nanmax([binned_stats['bin_edge_max'], binned_stats_blank['bin_edge_max']])])

            else:
                plt.ylim([np.nanmin(CI_lows), 1.2 * np.nanmax(CI_highs)])
                plt.xlim([binned_stats['bin_edge_min'], binned_stats['bin_edge_max']])
        else:
            for idepth, linecolor in zip(range(len(depth_list)), linecolors):
                if idepth == 0:
                    title_on = True
                else:
                    title_on = False
                plot_line_with_error(xarr=np.power(np.e, binned_stats['bin_centers'][idepth, :]),
                                     arr=binned_stats['bin_means'][idepth, :],
                                     CI_low=CI_lows[idepth], CI_high=CI_highs[idepth],
                                     linecolor=linecolor, label=str(int(depth_list[idepth] * 100)) + 'cm', marker='o-',
                                     markersize=3,
                                     xlabel=xlabel, ylabel='dF/F',
                                     title_on=title_on, title='', fontsize=fontsize, axis_fontsize=axis_fontsize,
                                     suffix='')
            if (len(trace_arr_blank) > 0) and (len(speed_arr_blank) > 0):
                binned_stats_blank = get_binned_stats(xarr=speed_arr_blank, yarr=trace_arr_blank, bin_number=speed_bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    CI_low, CI_high = get_confidence_interval(arr=[],
                                                              sem=np.divide(binned_stats_blank['bin_stds'][0], np.sqrt(
                                                                  binned_stats_blank['bin_counts'][0])),
                                                              sig_level=0.05,
                                                              mean_arr=binned_stats_blank['bin_means'][0])

                plot_line_with_error(xarr=np.power(np.e, binned_stats_blank['bin_centers'][0, :]),
                                     arr=binned_stats_blank['bin_means'][0, :],
                                     CI_low=CI_low, CI_high=CI_high,
                                     linecolor='gray', label='blank', marker='o-', markersize=3,
                                     xlabel=xlabel, ylabel='dF/F',
                                     title_on=False, title='', fontsize=fontsize, axis_fontsize=axis_fontsize,
                                     suffix='')
                plt.ylim([np.nanmin(np.concatenate((CI_lows.flatten(), CI_low.flatten()))),
                          1.2 * np.nanmax(np.concatenate((CI_highs.flatten(), CI_high.flatten())))])
                plt.xlim([np.power(np.e, np.nanmin([binned_stats['bin_edge_min'], binned_stats_blank['bin_edge_min']])),
                          np.power(np.e,
                                   np.nanmax([binned_stats['bin_edge_max'], binned_stats_blank['bin_edge_max']]))])

            else:
                plt.ylim([np.nanmin(CI_lows), 1.2 * np.nanmax(CI_highs)])
                plt.xlim([np.power(np.e, binned_stats['bin_edge_min']), np.power(np.e, binned_stats['bin_edge_max'])])
            ax = plt.gca()
            ax.set_xscale('log')

        plt.title(title, fontsize=fontsize)
        plt.legend(fontsize=10)

    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    return xlim, ylim


def get_binned_stats_2d(xarr1, xarr2, yarr, bin_number=None, bin_size=None, bin_edges=[], log=False, log_base=np.e):
    if len(bin_edges) > 0:
        binned_stats = {
            'bin_edge_min': [],
            'bin_edge_max': [],
            'bins': [],
            'bin_means': [],
            #             'bin_sems':np.zeros((bin_number-1,bin_number-1)),
            #             'bin_counts':np.zeros((bin_number-1,bin_number-1)),
            'bin_centers': [],
            'bin_edges': [],
            'bin_numbers': []
        }
        if log == True:
            bin_edges[0] = np.array(bin_edges[0])
            bin_edges[1] = np.array(bin_edges[1])
        binned_stats['bin_edge_min'].append(np.nanmin(bin_edges[0]))
        binned_stats['bin_edge_min'].append(np.nanmin(bin_edges[1]))
        binned_stats['bin_edge_max'].append(np.nanmax(bin_edges[0]))
        binned_stats['bin_edge_max'].append(np.nanmax(bin_edges[1]))
        binned_stats['bins'] = bin_edges
        binned_stats['bin_edges'].append(bin_edges[0])
        binned_stats['bin_edges'].append(bin_edges[1])
        binned_stats['bin_means'], _, _, _ = scipy.stats.binned_statistic_2d(xarr1.flatten(), xarr2.flatten(),
                                                                             yarr.flatten(), statistic='mean',
                                                                             bins=binned_stats['bins'],
                                                                             expand_binnumbers=True)

        for iaxis in range(2):
            bin_width = (binned_stats['bin_edges'][iaxis][1] - binned_stats['bin_edges'][iaxis][0])
            binned_stats['bin_centers'].append(binned_stats['bin_edges'][iaxis][1:] - bin_width / 2) #THIS IS WRONG!! THIS DOESN'T TAKE INTO ACCOUNT THE LOGGED NUMBER
            binned_stats['bin_numbers'].append(len(binned_stats['bin_edges'][iaxis][1:] - bin_width / 2))

    return binned_stats


def set_RS_OF_heatmap_axis_ticks(binned_stats, log=True):
    bin_numbers = binned_stats['bin_numbers']
    # bin_centers1 = np.array(np.power(log_base, binned_stats['bin_centers'][0]), dtype='object')
    # bin_centers2 = np.array((np.power(log_base, binned_stats['bin_centers'][1])),
    #                         dtype='object')
    bin_edges1 = np.array(binned_stats['bin_edges'][0], dtype='object')
    bin_edges2 = np.array(binned_stats['bin_edges'][1], dtype='object')
    ctr = 0
    for it in bin_edges1:
        if it >= 1:
            bin_edges1[ctr] = int(np.round(it))
        else:
            bin_edges1[ctr] = np.round(it, 2)
        ctr += 1
    ctr = 0
    for it in bin_edges2:
        if it >= 1:
            bin_edges2[ctr] = int(np.round(it))
        else:
            bin_edges2[ctr] = np.round(it, 2)
        ctr += 1
    if log == False:
        _, _ = plt.xticks(np.arange(bin_numbers[0]), bin_centers1, rotation=60, ha='center',
                          fontsize=15)
        _, _ = plt.yticks(np.arange(bin_numbers[1]), bin_center2, fontsize=15)
    else:
        ticks_select1 = (np.arange(-1, bin_numbers[0] * 2, 1) / 2)[0::2]
        ticks_select2 = (np.arange(-1, bin_numbers[1] * 2, 1) / 2)[0::2]
        _, _ = plt.xticks(ticks_select1, bin_edges1, rotation=60, ha='center', fontsize=15)
        _, _ = plt.yticks(ticks_select2, bin_edges2, fontsize=15)


def plot_RS_OF_heatmap(binned_stats, log=True, log_base=10,
                       xlabel='Running Speed (cm/s)', ylabel='Optic Flow (degree/s)', vmin=None, vmax=None):
    plt.imshow(binned_stats['bin_means'].T, cmap='Reds', origin='lower', vmin=vmin, vmax=vmax)
    set_RS_OF_heatmap_axis_ticks(binned_stats, log=log)
    plt.colorbar()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)


def get_mutual_info(arr1, arr2):
    arr1_mask = ~np.isnan(arr1)
    arr2_mask = ~np.isnan(arr2)
    mutual_mask = arr1_mask * arr2_mask
    arr1 = arr1[mutual_mask]
    arr2 = arr2[mutual_mask]
    mutual_info = mutual_info_score(arr1, arr2)

    return mutual_info


def plot_frame_off():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# Make an array for ROI mask for each ROI
def stats_to_array(stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False):
    """
    converts stats sequence of dictionaries to an array
    :param stats: sequence of dictionaries from stat.npy
    :param Ly: number of pixels along dim Y from ops dictionary
    :param Lx: number of pixels along dim X
    :param label_id: keeps ROI indexing
    :return: numpy stack of arrays, each containing x and y pixels for each ROI
    """
    arrays = []
    for i, stat in enumerate(stats):
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[stat['ypix'], stat['xpix']] = 1
        if label_id:
            arr *= i + 1
        arrays.append(arr)
    return (np.stack(arrays))


def find_roi_center(cells_mask, roi):
    #     cells_mask[cells_mask!=0] = 1
    #     cells_mask[cells_mask==0] = np.nan
    x_min = np.min(np.where(cells_mask[roi] > 0)[0])
    x_max = np.max(np.where(cells_mask[roi] > 0)[0])
    y_min = np.min(np.where(cells_mask[roi] > 0)[1])
    y_max = np.max(np.where(cells_mask[roi] > 0)[1])
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    radius = (x_range + y_range) / 2 / 2

    return int(x_center), int(y_center), int(radius)


def gaussian_func(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


