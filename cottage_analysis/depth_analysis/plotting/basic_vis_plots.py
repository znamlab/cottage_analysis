import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from typing import Sequence, Dict, Any
import scipy
from cottage_analysis.depth_analysis.plotting.plotting_utils import *
from cottage_analysis.depth_analysis.depth_preprocess.process_params import *

# --- Raster plot for different depths (running speed or dFF) --- #
def plot_raster_all_depths(values, dffs, depth_list, img_VS, stim_dict, distance_bins, plot_rows, plot_cols, which_row, which_col, heatmap_cmap, fontsize_dict, is_trace=True, roi=0, title='', frame_rate=15, distance_max=6, vmax=None, landscape=False):
    if is_trace:
        values_arr,_ = create_trace_arr_per_roi(roi, dffs, depth_list, stim_dict,
                                                mode='sort_by_depth', protocol='fix_length',
                                                blank_period=0, frame_rate=frame_rate)
        unit_scale = 1
    else:
        values_arr, _ = create_speed_arr(values, depth_list, stim_dict, mode='sort_by_depth', protocol='fix_length',
                                blank_period=0, frame_rate=frame_rate)
        unit_scale = 100
    distance_arr, _ = create_speed_arr(img_VS['EyeZ'], depth_list, stim_dict, mode='sort_by_depth',
                                    protocol='fix_length', blank_period=0, frame_rate=frame_rate)
    for idepth in range(distance_arr.shape[0]):
        for itrial in range(distance_arr.shape[1]):
            distance_arr[idepth, itrial, :] = distance_arr[idepth, itrial, :] - distance_arr[idepth, itrial, 0]
            
    binned_stats = get_binned_arr(xarr=distance_arr, yarr=values_arr, bin_number=distance_bins,
                                bin_edge_min=0, bin_edge_max=6)
    if vmax==None:
        vmax=np.nanmax(binned_stats['binned_yrr']) * unit_scale
    else:
        vmax=vmax
    for idepth in range(0, len(depth_list)):
        depth = depth_list[idepth]
        if landscape:
            plt.subplot2grid([plot_rows, plot_cols], [which_row, which_col+idepth])
        else: 
            plt.subplot2grid([plot_rows, plot_cols], [which_row+idepth, which_col])
        if idepth == 0:
            title_on = True
        else:
            title_on = False
        title = f'ROI {roi} {title}, \n'
        plot_raster(arr=np.array(binned_stats['binned_yrr'][idepth]) * unit_scale, vmin=0,
                    vmax=vmax,
                    cmap=heatmap_cmap, title=title, title_on=title_on,
                    suffix='Depth: ' + str(int(depth_list[idepth] * 100)) + ' cm', fontsize=fontsize_dict['title'],
                    frame_rate=frame_rate,
                    extent=[0, distance_max * 100, binned_stats['binned_yrr'][idepth].shape[0], 1],
                    set_nan_cmap=False)

        plt.xlabel('Distance (cm)', fontsize=fontsize_dict['xlabel'])
        plt.ylabel('Trial no.', fontsize=fontsize_dict['ylabel'])
        plt.xticks(fontsize=fontsize_dict['xticks'])
        plt.yticks(fontsize=fontsize_dict['yticks'])
    return binned_stats


# --- PSTH --- #
def plot_PSTH(values, dffs,  depth_list, img_VS, stim_dict, distance_bins, line_colors, heatmap_cmap, fontsize_dict, is_trace=True, roi=0, ylim=None, title='', frame_rate=15, distance_max=6):
    if is_trace:
        values_arr,_ = create_trace_arr_per_roi(roi, dffs, depth_list, stim_dict,
                                                mode='sort_by_depth', protocol='fix_length',
                                                blank_period=0, frame_rate=frame_rate)
        unit_scale = 1
    else:
        values_arr, _ = create_speed_arr(values, depth_list, stim_dict, mode='sort_by_depth', protocol='fix_length',
                                blank_period=0, frame_rate=frame_rate)
        unit_scale = 100 # scale depth unit from m to cm
    distance_arr, _ = create_speed_arr(img_VS['EyeZ'], depth_list, stim_dict, mode='sort_by_depth',
                                    protocol='fix_length', blank_period=0, frame_rate=frame_rate)
    for idepth in range(distance_arr.shape[0]):
        for itrial in range(distance_arr.shape[1]):
            distance_arr[idepth, itrial, :] = distance_arr[idepth, itrial, :] - distance_arr[idepth, itrial, 0]
            
    binned_stats = get_binned_arr(xarr=distance_arr, yarr=values_arr, bin_number=distance_bins,
                                bin_edge_min=0, bin_edge_max=6)

    for idepth, linecolor in zip(range(len(depth_list)), line_colors):
        CI_low, CI_high = get_confidence_interval(binned_stats['binned_yrr'][idepth] * unit_scale,
                                                    mean_arr=np.nanmean(binned_stats['binned_yrr'][idepth] * unit_scale,
                                                                        axis=0))
        plot_line_with_error(xarr=np.linspace(0, distance_max * unit_scale, distance_bins),
                                arr=np.nanmean(binned_stats['binned_yrr'][idepth] * unit_scale, axis=0),
                                CI_low=CI_low, CI_high=CI_high, linecolor=linecolor,
                                label=str(int(depth_list[idepth] * 100)) + ' cm', fontsize=fontsize_dict['title'], linewidth=3)
        plt.legend(fontsize=fontsize_dict['legend'], framealpha=0.3)
        plt.title('Running Speed (cm/s)', fontsize=fontsize_dict['title'])
    plt.xlabel('Distance (cm)', fontsize=fontsize_dict['xlabel'])
    plt.ylabel('Running speed (cm/s)', fontsize=fontsize_dict['ylabel'])
    plt.xticks(fontsize=fontsize_dict['xticks'])
    plt.yticks(fontsize=fontsize_dict['yticks'])
    xlim = plt.gca().get_xlim()
    if ylim==None:
        ylim = [-0.2, plt.gca().get_ylim()[1]]
        plt.ylim(ylim)
    else: 
        ylim=ylim
        plt.ylim(ylim)
    despine()
        
    return binned_stats, xlim, ylim


# --- Depth tuning curve --- #
MIN_SIGMA=0.5
def gaussian_func(x, a, x0, log_sigma,b):
    a = a
    sigma = np.exp(log_sigma)+MIN_SIGMA
    return (a * np.exp(-(x - x0) ** 2) / (2 * sigma ** 2))+b

def plot_depth_tuning_curve(dffs, speeds, roi, speed_thr_cal, depth_list, stim_dict, depth_neurons, gaussian_depth, fontsize_dict, this_depth=None, ylim=None, frame_rate=15):
    trace_arr_noblank, _ = create_trace_arr_per_roi(which_roi=roi, 
                                                    dffs=dffs,
                                                    depth_list=depth_list, 
                                                    stim_dict=stim_dict,
                                                    mode='sort_by_depth', 
                                                    protocol='fix_length',
                                                    blank_period=0, 
                                                    frame_rate=frame_rate)
    speed_arr_noblank, _ = create_speed_arr(speeds=speeds, 
                                            depth_list=depth_list, 
                                            stim_dict=stim_dict,
                                            mode='sort_by_depth',
                                            protocol='fix_length', 
                                            blank_period=0,
                                            frame_rate=frame_rate)
    
    trace_arr_noblank[speed_arr_noblank < speed_thr_cal] = np.nan
    trace_arr_mean_eachtrial = np.nanmean(trace_arr_noblank, axis=2)
    CI_lows = np.zeros(len(depth_list))
    CI_highs = np.zeros(len(depth_list))
    for idepth in range(len(depth_list)):
        CI_lows[idepth], CI_highs[idepth] = get_confidence_interval(
            trace_arr_mean_eachtrial[idepth, :],
            mean_arr=np.nanmean(trace_arr_mean_eachtrial, axis=1)[idepth].reshape(-1, 1))

    if (this_depth == None) or (this_depth!=len(depth_list)): # we can't plot the tuning curve for non-depth-selective neurons
        plot_line_with_error(arr=np.nanmean(trace_arr_mean_eachtrial, axis=1), CI_low=CI_lows,
                             CI_high=CI_highs, linecolor='b', fontsize=fontsize_dict['title'], linewidth=3)

        trace_arr_mean_eachtrial = np.nanmean(trace_arr_noblank, axis=2)
        x = np.log(np.repeat(np.array(depth_list), trace_arr_mean_eachtrial.shape[1]))
        roi_number = np.where(depth_neurons == roi)[0][0]
        [a, x0, log_sigma, b] = gaussian_depth[gaussian_depth.ROI==roi][['a','x0_logged','log_sigma','b']].values[0].astype('float32')
        plt.plot(np.linspace(0, len(depth_list) - 1, 100),
                    gaussian_func(np.linspace(np.log(depth_list[0]*100), np.log(depth_list[-1]*100), 100), a,
                                x0, log_sigma,b), 'gray', linewidth=3)

        plt.xticks(np.arange(len(depth_list)), (np.array(depth_list) * 100).astype('int'),
                   fontsize=fontsize_dict['xticks'])
        plt.yticks(fontsize=fontsize_dict['yticks'])
        plt.ylabel('dF/F', fontsize=fontsize_dict['ylabel'])
        plt.xlabel('Depth (cm/s)', fontsize=fontsize_dict['xlabel'])
        plt.title('Depth tuning (Closeloop)', fontsize=fontsize_dict['title'])
        ylim = [0,plt.gca().get_ylim()[1]]
        despine()
    else:
        ylim = None

    plot_line_with_error(arr=np.nanmean(trace_arr_mean_eachtrial, axis=1), CI_low=CI_lows,
                         CI_high=CI_highs, linecolor='b', fontsize=fontsize_dict['title'], linewidth=3)
    plt.xticks(np.arange(len(depth_list)), (np.array(depth_list) * 100).astype('int'),
               fontsize=fontsize_dict['xticks'])
    plt.ylabel('dF/F', fontsize=fontsize_dict['ylabel'])
    plt.xlabel('Depth (cm)', fontsize=fontsize_dict['xlabel'])
    plt.title('Depth tuning (CloseLoop)', fontsize=fontsize_dict['title'])
    if ylim != None:
        plt.ylim(ylim)
    else:
        ylim = plt.gca().get_ylim()
    despine()
    
    return ylim


# --- RS/OF - trace heatmap --- #
def get_RS_OF_heatmap_matrix(speeds, optics, roi, dffs, depth_list, img_VS, stim_dict, log_range, playback=False, speed_thr=0.01, of_thr=0.03, frame_rate=15):
    extended_matrix = np.zeros((log_range['rs_bin_num'],log_range['of_bin_num']))
    
    # calculate all RS/OF arrays
    speed_arr_noblank, _ = create_speed_arr(speeds, depth_list, stim_dict,
                                                    mode='sort_by_depth',
                                                    protocol='fix_length', blank_period=0,
                                                    frame_rate=frame_rate)
    frame_num_pertrial_max_playback = speed_arr_noblank.shape[2]
    total_trials = speed_arr_noblank.shape[1]

    of_arr_noblank, _ = create_speed_arr(optics, depth_list, stim_dict, mode='sort_by_depth',
                                                protocol='fix_length', blank_period=0,
                                                frame_rate=frame_rate)
    trace_arr_noblank, _ = create_trace_arr_per_roi(roi, dffs, depth_list, stim_dict,
                                                            mode='sort_by_depth', protocol='fix_length',
                                                            blank_period=0, frame_rate=frame_rate)
    
    if playback:
        # When RS = 0:
        speeds_playback_unthred = img_VS.MouseZ.diff() / img_VS.HarpTime.diff()  # with no playback. EyeZ and MouseZ should be the same.
        speeds_playback_unthred[0] = 0
        speeds_playback_unthred[speeds_playback_unthred < 0] = 0
        speed_arr_playback_unthred, _ = create_speed_arr(speeds_playback_unthred, depth_list, stim_dict,
                                                            mode='sort_by_depth', protocol='fix_length',
                                                            blank_period=0, frame_rate=frame_rate)
        speeds_eye_playback_unthred = img_VS.EyeZ.diff() / img_VS.HarpTime.diff()  # EyeZ is how the perspective of animal moves
        speeds_eye_playback_unthred[0] = 0
        speeds_playback_unthred[speeds_playback_unthred < 0] = 0
        optics_playback_unthred= calculate_OF(rs=speeds_eye_playback_unthred, img_VS=img_VS, mode='no_RF')
        of_arr_playback_unthred, _ = create_speed_arr(optics_playback_unthred, depth_list, stim_dict,
                                                                mode='sort_by_depth',
                                                                protocol='fix_length', blank_period=0,
                                                                frame_rate=frame_rate)

        zero_idx_rs = np.where(speed_arr_playback_unthred <= speed_thr)
        speeds_playback_zeros_rs = np.array(
            [speed_arr_playback_unthred[zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]] for i in range(len(zero_idx_rs[0]))])
        of_playback_zeros_rs = np.array(
            [of_arr_playback_unthred[zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]] for i in range(len(zero_idx_rs[0]))])
        trace_playback_zeros_rs = np.array(
            [trace_arr_noblank[zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]] for i in range(len(zero_idx_rs[0]))])

        
        # When OF = 0:
        zero_idx_of = np.where(of_arr_playback_unthred <= of_thr)
        speeds_playback_zeros_of = np.array(
            [speed_arr_playback_unthred[zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]] for i in range(len(zero_idx_of[0]))])
        of_playback_zeros_of = np.array(
            [of_arr_playback_unthred[zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]] for i in range(len(zero_idx_of[0]))])
        trace_playback_zeros_of = np.array(
            [trace_arr_noblank[zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]] for i in range(len(zero_idx_of[0]))])

        
    # Matrix binned by RS & OF
    binned_stats = get_binned_stats_2d(xarr1=thr(speed_arr_noblank * 100, speed_thr * 100),
                                    xarr2=np.degrees(of_arr_noblank),
                                    yarr=trace_arr_noblank,
                                    bin_edges=[np.logspace(log_range['rs_bin_log_min'], log_range['rs_bin_log_max'],
                                                            num=log_range['rs_bin_num'], base=log_range['log_base'])
                                        , np.logspace(log_range['of_bin_log_min'], log_range['of_bin_log_max'],
                                                        num=log_range['of_bin_num'], base=log_range['log_base'])],
                                    log=True, log_base=log_range['log_base'])
    
    vmin_heatmap_closeloop = np.nanmin(binned_stats['bin_means'])
    vmax_heatmap_closeloop = np.nanmax(binned_stats['bin_means'])
    
    extended_matrix[1:,1:] = binned_stats['bin_means']
    
    if playback:
        binned_stats_zeros_rs = get_binned_stats_2d(xarr1=speeds_playback_zeros_rs * 100,
                                        xarr2=np.degrees(of_playback_zeros_rs),
                                        yarr=trace_playback_zeros_rs,
                                        bin_edges=[np.array([0, 1])
                                            , np.logspace(log_range['of_bin_log_min'], log_range['of_bin_log_max'],
                                                            num=log_range['of_bin_num'], base=log_range['log_base'])],
                                        log=True, log_base=log_range['log_base'])
        
        binned_stats_zeros_of = get_binned_stats_2d(xarr1=speeds_playback_zeros_of * 100,
                                        xarr2=np.degrees(of_playback_zeros_of),
                                        yarr=trace_playback_zeros_of,
                                        bin_edges=[np.logspace(log_range['rs_bin_log_min'], log_range['rs_bin_log_max'],
                                                            num=log_range['rs_bin_num'], base=log_range['log_base']),
                                                   np.array([0, 1])],
                                        log=True, log_base=log_range['log_base'])
        
        extended_matrix[0,1:] = binned_stats_zeros_rs['bin_means'].flatten()
        extended_matrix[1:,0] = binned_stats_zeros_of['bin_means'].flatten()
    
    return extended_matrix


def set_RS_OF_heatmap_axis_ticks(log_range, fontsize_dict, playback=False, log=True):
    bin_numbers = [log_range['rs_bin_num']-1,log_range['of_bin_num']-1]
    bin_edges1 = np.logspace(log_range['rs_bin_log_min'], log_range['rs_bin_log_max'],
                                                            num=log_range['rs_bin_num'], base=log_range['log_base'])
    bin_edges2 = np.logspace(log_range['of_bin_log_min'], log_range['of_bin_log_max'],
                                                            num=log_range['of_bin_num'], base=log_range['log_base'])
    if playback:
        bin_numbers = [log_range['rs_bin_num'],log_range['of_bin_num']]
        bin_edges1 = np.insert(bin_edges1, 0, 0)
        bin_edges2 = np.insert(bin_edges2, 0, 0)
    bin_edges1 = bin_edges1.tolist()
    bin_edges2 = bin_edges2.tolist()
    ctr = 0
    for it in bin_edges1:
        if (it >= 1) or (it==0):
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
    # if log == False:
    #     _, _ = plt.xticks(np.arange(bin_numbers[0]), bin_centers1, rotation=60, ha='center',
    #                       fontsize=fontsize_dict['xticks'])
    #     _, _ = plt.yticks(np.arange(bin_numbers[1]), bin_centers2, fontsize=fontsize_dict['yticks'])
    else:
        ticks_select1 = ((np.arange(-1, bin_numbers[0] * 2, 1) / 2)[0::2])
        ticks_select2 = ((np.arange(-1, bin_numbers[1] * 2, 1) / 2)[0::2])
        _, _ = plt.xticks(ticks_select1, bin_edges1, rotation=60, ha='center', fontsize=fontsize_dict['xticks'])
        _, _ = plt.yticks(ticks_select2, bin_edges2, fontsize=fontsize_dict['yticks'])
   


def plot_RS_OF_heatmap_extended(matrix, log_range, playback, fontsize_dict, log=True,
                       xlabel='Running Speed (cm/s)', ylabel='Optic Flow (degree/s)', vmin=None, vmax=None):
    if not playback:
        matrix = matrix[1:,1:]
    plt.imshow(matrix.T, cmap='Reds', origin='lower', vmin=vmin, vmax=vmax)
    set_RS_OF_heatmap_axis_ticks(log_range=log_range, fontsize_dict=fontsize_dict, playback=playback, log=log)
    plt.colorbar()
    plt.xlabel(xlabel, fontsize=fontsize_dict['xlabel'])
    plt.ylabel(xlabel, fontsize=fontsize_dict['ylabel'])     
        


