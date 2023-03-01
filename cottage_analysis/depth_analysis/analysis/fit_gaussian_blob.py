from typing import Any

import os
import sys
import defopt
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from suite2p.extraction import masks
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # save text as text not outlines
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings


import cottage_analysis as cott
from cottage_analysis.depth_analysis.filepath.generate_filepaths import *
from cottage_analysis.imaging.common import find_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
from cottage_analysis.stimulus_structure import sphere_structure as vis_stim_structure
from cottage_analysis.depth_analysis.plotting.plotting_utils import *
from cottage_analysis.depth_analysis.depth_preprocess.process_params import *
from cottage_analysis.depth_analysis.depth_preprocess.process_trace import *

from cottage_analysis.io_module import harp

import flexiznam as flz
from pathlib import Path
from warnings import warn
from flexiznam.schema import Dataset


MIN_SIGMA=0.1
def twoD_Gaussian(xy_tuple, log_amplitude, xo, yo, log_sigma_x2, log_sigma_y2, theta, offset):
    (x,y) = xy_tuple
    sigma_x_sq = np.exp(log_sigma_x2) + MIN_SIGMA # 0.25
    sigma_y_sq = np.exp(log_sigma_y2) + MIN_SIGMA  # 0.25
    amplitude = np.exp(log_amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x_sq) + (np.sin(theta)**2)/(2*sigma_y_sq)
    b = (np.sin(2*theta))/(4*sigma_x_sq) - (np.sin(2*theta))/(4*sigma_y_sq)
    c = (np.sin(theta)**2)/(2*sigma_x_sq) + (np.cos(theta)**2)/(2*sigma_y_sq)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            +c*((y-yo)**2)))
    return g


def gaussian_blob_plotting(roi, speed_arr_noblank, speed_thr, of_arr_noblank, trace_arr_noblank, popt, Z_fit, X, Y, r_sq, Z_, Z_fit_):
    rs_bin_log_min = 0
    rs_bin_log_max = 2.5
    rs_bin_num = 6
    of_bin_log_min = -1.5
    of_bin_log_max = 3.5
    of_bin_num = 11
    log_base = 10

    plt.figure(figsize=(15,5))
    # Heatmap before
    binned_stats = get_binned_stats_2d(xarr1=thr(speed_arr_noblank * 100, speed_thr * 100),
                                       xarr2=np.degrees(of_arr_noblank),
                                       yarr=trace_arr_noblank,
                                       bin_edges=[np.logspace(rs_bin_log_min, rs_bin_log_max,
                                                              num=rs_bin_num, base=log_base)
                                           , np.logspace(of_bin_log_min, of_bin_log_max,
                                                         num=of_bin_num, base=log_base)],
                                       log=True, log_base=log_base)
    vmin = np.nanmin(binned_stats['bin_means'])
    vmax = np.nanmax(binned_stats['bin_means'])
    plt.subplot(141)
    plot_RS_OF_heatmap(binned_stats=binned_stats, log=True, log_base=10,
                       vmin = vmin, vmax = vmax,
                       xlabel='Running Speed (cm/s)', ylabel='Optic Flow (degree/s)')
    plt.title('ROI'+str(roi)+' Original heatmap')

    trace_arr_noblank_fit = Z_fit.reshape(X.shape)
    binned_stats = get_binned_stats_2d(xarr1=thr(speed_arr_noblank * 100, speed_thr * 100),
                                       xarr2=np.degrees(of_arr_noblank),
                                       yarr=trace_arr_noblank_fit,
                                       bin_edges=[np.logspace(rs_bin_log_min, rs_bin_log_max,
                                                              num=rs_bin_num, base=log_base)
                                           , np.logspace(of_bin_log_min, of_bin_log_max,
                                                         num=of_bin_num, base=log_base)],
                                       log=True, log_base=log_base)
    plt.subplot(142)
    plot_RS_OF_heatmap(binned_stats=binned_stats, log=True, log_base=10,
                       vmin = vmin, vmax = vmax,
                       xlabel='Running Speed (cm/s)', ylabel='Optic Flow (degree/s)')
    plt.title(f'Fitted heatmap, theta {np.around(np.degrees(popt[5]))}, \nR2 {np.round(r_sq,3)}')

    # Gaussian kernel
    plt.subplot(143)
    n = 10
    x = np.log(np.logspace(rs_bin_log_min, rs_bin_log_max, num=rs_bin_num * n, base=log_base))
    y = np.log((np.logspace(of_bin_log_min, of_bin_log_max, num=of_bin_num * n, base=log_base)))
    xx_veri, yy_veri = np.meshgrid(x, y)
    z_veri = twoD_Gaussian(np.array([xx_veri.T.flatten(), yy_veri.T.flatten()]), *popt)
    plt.imshow(z_veri.reshape(rs_bin_num * n, of_bin_num * n).T, origin='lower',
               aspect='equal', interpolation=None, cmap='Reds', vmax=vmax)
    plt.colorbar()
    plt.title('Gaussian kernel')
    plt.axis('off')

    # Plot actual vs fit trace
    plt.subplot(144)
    plt.plot(Z_, label='original')
    plt.plot(Z_fit_, label='fitted')
    plt.legend()
    plt.tight_layout(pad=1)
    
    
def main(project, mouse, session):
    """
    :param str project: project name (determines the root directory for raw data)
    :param str mouse: mouse name
    :param str session: session name,Sdate
    :return: None
    """
    analysis = True
    plot=True
    
    flexilims_session = flz.get_flexilims_session(project_id=project)
    sess_children = get_session_children(project=project, mouse=mouse, session=session, flexilims_session=flexilims_session)
    if len(sess_children[sess_children.name.str.contains('Playback')])>0:
        protocols = ['SpheresPermTubeReward','SpheresPermTubeRewardPlayback']
    else:
        protocols = ['SpheresPermTubeReward']
    root = Path(flz.PARAMETERS["data_root"]["processed"])

    # ----- SETUPS -----
    frame_rate = 15
    speed_thr_cal = 0.2  # m/s, threshold for running speed when calculating depth neurons
    # depth_list = [0.06, 0.19, 0.6, 1.9, 6]
    # choose_trials = 50
    speed_thr = 0.01  # m/s

    manual_choose_rois = False
    manually_chosen_rois = [0, 22, 27, 107, 112, 157]

    # fit setup
    rs_min = 0.5
    rs_max = 500
    of_min = 0.03
    of_max = 3000
    batch_num = 5

    for protocol in protocols:
        print(f'---------Process protocol {protocol}/{len(protocols)}---------', flush=True)
        # ----- STEP1: Generate file path -----
        session_analysis_folder_original = root/project/'Analysis'/mouse/session/protocols[0]
        session_protocol_folder = root/project/mouse/session/protocol
        session_analysis_folder = root/project/'Analysis'/mouse/session/protocol
        (
        _,
        _,
        _,
        suite2p_folder,
        _,
        ) = generate_file_folders(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            all_protocol_recording_entries=None, 
            recording_no=0, 
            flexilims_session=None
        )

        print('---STEP 1 FINISHED.---', '\n', flush=True)

        # ----- STEP2: Load files -----
        print('---START STEP 2---', '\n', 'Load files...', flush=True)
        # Load suite2p files
        ops = np.load(suite2p_folder/'ops.npy', allow_pickle=True)
        ops = ops.item()
        iscell = np.load(suite2p_folder/'iscell.npy', allow_pickle=True)[:, 0]
        # F = np.load(trace_folder + 'F.npy', allow_pickle=True)
        # Fast = np.load(trace_folder + 'Fast.npy', allow_pickle=True)
        # # Fneu = np.load(trace_folder + 'Fneu.npy', allow_pickle=True)
        # # spks = np.load(trace_folder + 'spks.npy', allow_pickle=True)

        # All_rois
        which_rois = (np.arange(iscell.shape[0]))[iscell.astype('bool')]

        with open(session_protocol_folder/"plane0/img_VS_all.pickle", "rb") as handle:
            img_VS_all = pickle.load(handle)
        with open(session_protocol_folder/"plane0/stim_dict_all.pickle", "rb") as handle:
            stim_dict_all = pickle.load(handle)
        dffs_ast_all = np.load(session_protocol_folder/"plane0/dffs_ast_all.npy")
        depth_neurons = np.load(session_analysis_folder_original/"plane0/depth_neurons.npy")
        max_depths = np.load(session_analysis_folder_original/'plane0/max_depths_index.npy')
        with open(session_analysis_folder/('plane0/gaussian_depth_tuning_fit_new_'+str(0.5)+'.pickle'), 'rb') as handle:
            gaussian_depth_fit_df = pickle.load(handle)

        depth_list = img_VS_all["Depth"].unique()
        depth_list = np.round(depth_list, 2)
        depth_list = depth_list[~np.isnan(depth_list)].tolist()
        depth_list.remove(-99.99)
        depth_list.sort()

        # ----- STEP3: Process params -----
        if 'Playback' in protocol:
            is_actual_running_list = [True, False]
        else:
            is_actual_running_list = [True]
        for is_actual_running in is_actual_running_list:
            print('---START STEP 3---', '\n', 'Process params...', flush=True)
            # Running speed
            # Running speed is thresholded with a small threshold to get rid of non-zero values (default threshold 0.01)
            # speeds = img_VS.MouseZ.diff() / img_VS.HarpTime.diff() # with no playback. EyeZ and MouseZ should be the same.
            speeds = img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()  # CHANGE TO MOUSEZ AFTERWARDS!!!
            speeds[0] = 0
            speeds = thr(speeds, speed_thr)
            # speed_arr, _ = create_speed_arr(speeds, depth_list, stim_dict, mode='sort_by_depth', protocol='fix_length',
            #                                 blank_period=0, frame_rate=frame_rate)
            # speed_arr_mean = np.nanmean(speed_arr, axis=1)
            speed_arr_noblank, _ = create_speed_arr(speeds, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                    blank_period=0, frame_rate=frame_rate)
            speed_arr_noblank_mean = np.nanmean(speed_arr_noblank, axis=1)
            speed_arr_blank, _ = create_speed_arr(speeds, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                isStim=False, blank_period=0, frame_rate=frame_rate)
            frame_num_pertrial_max = speed_arr_noblank.shape[2]
            total_trials = speed_arr_noblank.shape[1]

            # OF (Unit: rad/s)
            if 'Playback' in protocol:
                speeds_eye = img_VS_all.EyeZ.diff() / img_VS_all.HarpTime.diff()  # EyeZ is how the perspective of animal moves
                speeds_eye[0] = 0
                speeds_eye = thr(speeds_eye, speed_thr)
                speed_eye_arr_noblank, _ = create_speed_arr(speeds_eye, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                    blank_period=0, frame_rate=frame_rate)
                optics = calculate_OF(rs=speeds_eye, img_VS=img_VS_all, mode='no_RF')

            else:
                optics = calculate_OF(rs=speeds, img_VS=img_VS_all, mode='no_RF')
            of_arr_noblank, _ = create_speed_arr(optics, depth_list, stim_dict_all, mode='sort_by_depth',
                                                protocol='fix_length', blank_period=0,
                                                frame_rate=frame_rate)
            print('---STEP 3 FINISHED.---', '\n', flush=True)
            
            
            # VER 2: FIT 2D GAUSSIAN TO LOG(RS) & LOG(OF) (Without plotting)
            # ----- STEP4: 2D Gaussian fit for dff-RS/OF -----
            # Fit to 2D gaussian, see equations from https://handwiki.org/wiki/Gaussian_function
            print('---START STEP 4---', '\n', 'Fit gaussian blob...', flush=True)
            print('MIN SIGMA', str(MIN_SIGMA), flush=True)
            if is_actual_running:
                print('Fit actual running speed...', flush=True)
                speed_arr = speed_arr_noblank
                X = speed_arr_noblank.flatten()
                sfx = '_actual_running'
            else:
                print('Fit virtual running speed...', flush=True)
                speed_arr = speed_eye_arr_noblank
                X = speed_eye_arr_noblank.flatten()
                sfx = '_virtual_running'
            Y = of_arr_noblank.flatten()

            results = pd.DataFrame(
                columns=['ROI', 'preferred_depth_idx', 'preferred_depth_gaussian', 'log_amplitude', 'xo_logged', 'yo_logged', 'log_sigma_x2', 'log_sigma_y2',
                        'theta', 'offset', 'r_sq'])
            results['ROI'] = depth_neurons
            results['preferred_depth_idx'] = max_depths
            results['preferred_depth_gaussian'] = np.exp(np.array(gaussian_depth_fit_df.x0_logged).astype('float64'))
            
            
            # Loop through all rois
            if analysis:
                for iroi, choose_roi in enumerate(depth_neurons):
                    roi = choose_roi
                    trace_arr_noblank, _ = create_trace_arr_per_roi(roi, dffs_ast_all, depth_list, stim_dict_all, mode='sort_by_depth',
                                                                    protocol='fix_length', blank_period=0, frame_rate=frame_rate)
                    Z = trace_arr_noblank.flatten()

                    Z_ = Z[~np.isnan(Z)]
                    X_ = X[~np.isnan(Z)]
                    Y_ = Y[~np.isnan(Z)]
                    log_X_ = np.log(X_ * 100)
                    log_Y_ = np.log(np.degrees(Y_))
                    popt_arr = []
                    rsq_arr = []
                    np.random.seed(42)
                    for ibatch in range(batch_num):
                        mu0 = 0
                        sigma0 = 1

                        p0 = np.concatenate((np.random.normal(mu0, sigma0, size=1),
                                                np.atleast_1d(
                                                    np.min([np.abs(np.random.normal(mu0, sigma0, size=1)), np.log(rs_min)])),
                                                np.atleast_1d(
                                                    [np.min([np.abs(np.random.normal(mu0, sigma0, size=1)) + np.log(of_min)])]),
                                                np.random.normal(mu0, sigma0, size=2),
                                                np.atleast_1d(
                                                    np.max([np.abs(np.random.normal(mu0, sigma0, size=1)), np.radians(90)])),
                                                np.random.normal(mu0, sigma0, size=1)))
                        popt, pcov = curve_fit(twoD_Gaussian, (log_X_, log_Y_), Z_, maxfev=100000,
                                                bounds=([-np.inf, np.log(rs_min), np.log(of_min), -np.inf, -np.inf, 0, -np.inf],
                                                        [np.inf, np.log(rs_max), np.log(of_max), np.inf, np.inf, np.radians(90),
                                                        np.inf])
                                                )

                        Z_fit_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt)
                        Z_fit = np.empty(Z.shape)
                        Z_fit[:] = np.NaN
                        Z_fit[~np.isnan(Z)] = Z_fit_
                        mse = np.mean((Z_fit_ - Z_) ** 2)
                        r_sq = calculate_R_squared(Z_, Z_fit_)
                        popt_arr.append(popt)
                        rsq_arr.append(r_sq)
                    idx_best = np.argmax(np.array(rsq_arr))
                    popt_best = popt_arr[idx_best]
                    Z_fit_best_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt_best)
                    Z_fit_best = np.empty(Z.shape)
                    Z_fit_best[:] = np.NaN
                    Z_fit_best[~np.isnan(Z)] = Z_fit_best_
                    mse = np.mean((Z_fit_best_ - Z_) ** 2)
                    r_sq_best = calculate_R_squared(Z_, Z_fit_best_)

                    results.iloc[iroi, 3:-1] = popt_best
                    results.iloc[iroi, -1] = r_sq_best
                    # print(str(roi), np.round(np.exp(popt_best[1:3]),2), flush=True)
                    if iroi % 10 == 0:
                        print(roi, flush=True)
                    iroi += 1

                save_filename = session_analysis_folder/('plane0/gaussian_blob_fit_new_'+str(MIN_SIGMA)+sfx+'.pickle')
                with open(save_filename, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('---STEP 4 FINISHED.---', '\n', flush=True)
            
            
    print('---Start plotting.---', '\n', flush=True)
    if plot:
        for protocol in protocols:
            print(f'---------Process protocol {protocol}/{len(protocols)}---------', flush=True)
            # ----- STEP1: Generate file path -----
            session_analysis_folder_original = root/project/'Analysis'/mouse/session/protocols[0]
            session_protocol_folder = root/project/mouse/session/protocol
            session_analysis_folder = root/project/'Analysis'/mouse/session/protocol
            (
            _,
            _,
            _,
            suite2p_folder,
            _,
            ) = generate_file_folders(
                project=project,
                mouse=mouse,
                session=session,
                protocol=protocol,
                all_protocol_recording_entries=None, 
                recording_no=0, 
                flexilims_session=None
            )


            # ----- STEP3: Process params -----
            if 'Playback' in protocol:
                is_actual_running_list = [True, False]
            else:
                is_actual_running_list = [True]
            for is_actual_running in is_actual_running_list:
                # Running speed
                # Running speed is thresholded with a small threshold to get rid of non-zero values (default threshold 0.01)
                # speeds = img_VS.MouseZ.diff() / img_VS.HarpTime.diff() # with no playback. EyeZ and MouseZ should be the same.
                speeds = img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()  # CHANGE TO MOUSEZ AFTERWARDS!!!
                speeds[0] = 0
                speeds = thr(speeds, speed_thr)
                # speed_arr, _ = create_speed_arr(speeds, depth_list, stim_dict, mode='sort_by_depth', protocol='fix_length',
                #                                 blank_period=0, frame_rate=frame_rate)
                # speed_arr_mean = np.nanmean(speed_arr, axis=1)
                speed_arr_noblank, _ = create_speed_arr(speeds, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                        blank_period=0, frame_rate=frame_rate)
                speed_arr_noblank_mean = np.nanmean(speed_arr_noblank, axis=1)
                speed_arr_blank, _ = create_speed_arr(speeds, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                    isStim=False, blank_period=0, frame_rate=frame_rate)
                frame_num_pertrial_max = speed_arr_noblank.shape[2]
                total_trials = speed_arr_noblank.shape[1]

                # OF (Unit: rad/s)
                if 'Playback' in protocol:
                    speeds_eye = img_VS_all.EyeZ.diff() / img_VS_all.HarpTime.diff()  # EyeZ is how the perspective of animal moves
                    speeds_eye[0] = 0
                    speeds_eye = thr(speeds_eye, speed_thr)
                    speed_eye_arr_noblank, _ = create_speed_arr(speeds_eye, depth_list, stim_dict_all, mode='sort_by_depth', protocol='fix_length',
                                                        blank_period=0, frame_rate=frame_rate)
                    optics = calculate_OF(rs=speeds_eye, img_VS=img_VS_all, mode='no_RF')

                else:
                    optics = calculate_OF(rs=speeds, img_VS=img_VS_all, mode='no_RF')
                of_arr_noblank, _ = create_speed_arr(optics, depth_list, stim_dict_all, mode='sort_by_depth',
                                                    protocol='fix_length', blank_period=0,
                                                    frame_rate=frame_rate)
                if is_actual_running:
                    print('Plot actual running speed...', flush=True)
                    speed_arr = speed_arr_noblank
                    X = speed_arr_noblank.flatten()
                    sfx = '_actual_running'
                else:
                    print('Plot virtual running speed...', flush=True)
                    speed_arr = speed_eye_arr_noblank
                    X = speed_eye_arr_noblank.flatten()
                    sfx = '_virtual_running'
                Y = of_arr_noblank.flatten()
                
                save_prefix = 'plane0/plots/gaussian_blob_fit_new_'+str(MIN_SIGMA)+sfx+'/'
                if not os.path.exists(session_analysis_folder/save_prefix):
                    os.makedirs(session_analysis_folder/save_prefix)
                with open(session_analysis_folder/('plane0/gaussian_blob_fit_new_'+str(MIN_SIGMA)+sfx+'.pickle'), 'rb') as handle:
                    results = pickle.load(handle)
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if manual_choose_rois:
                        select_rois = manually_chosen_rois
                        for choose_roi in select_rois:
                            roi = choose_roi
                            trace_arr_noblank, _ = create_trace_arr_per_roi(roi, dffs_ast_all, depth_list, stim_dict_all,
                                                                            mode='sort_by_depth',
                                                                            protocol='fix_length', blank_period=0,
                                                                            frame_rate=frame_rate)
                            Z = trace_arr_noblank.flatten()

                            Z_ = Z[~np.isnan(Z)]
                            X_ = X[~np.isnan(Z)]
                            Y_ = Y[~np.isnan(Z)]
                            log_X_ = np.log(X_ * 100)
                            log_Y_ = np.log(np.degrees(Y_))
                            popt = results[results.ROI==roi].values[0][3:-1]
                            r_sq = results[results.ROI==roi]['r_sq'].values[0]
                            Z_fit_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt)
                            Z_fit = np.empty(Z.shape)
                            Z_fit[:] = np.NaN
                            Z_fit[~np.isnan(Z)] = Z_fit_
                            gaussian_blob_plotting(roi=roi, speed_arr_noblank=speed_arr, speed_thr=speed_thr,
                                                    of_arr_noblank=of_arr_noblank, trace_arr_noblank=trace_arr_noblank,
                                                    popt=popt, Z_fit=Z_fit, X=X, Y=Y, r_sq=r_sq, Z_=Z_, Z_fit_=Z_fit_)

                            if not os.path.exists(session_analysis_folder/save_prefix/'examples/'):
                                os.makedirs(session_analysis_folder/save_prefix/'examples/')

                            plt.savefig(session_analysis_folder/save_prefix/('examples/roi' + str(choose_roi) + '.pdf'))
                            print('ROI' + str(choose_roi), flush=True)

                    else:
                        for this_depth in range(len(depth_list)):

                            if not os.path.exists(
                                    session_analysis_folder/save_prefix/('depth' + str(depth_list[this_depth]) + '/')):
                                os.makedirs(session_analysis_folder/save_prefix/('depth' + str(depth_list[this_depth]) + '/'))
                            save_folder = session_analysis_folder/save_prefix/('depth' + str(depth_list[this_depth]) + '/')

                            neurons_this_depth = depth_neurons[max_depths == this_depth]
                            if len(neurons_this_depth) > 0:
                                rois_mins, rois_maxs = segment_arr(np.arange(len(neurons_this_depth)), segment_size=10)
                                for rois_min, rois_max in zip(rois_mins, rois_maxs):
                                    if rois_min != rois_max:
                                        select_rois = np.array(neurons_this_depth[rois_min:rois_max])
                                    else:
                                        select_rois = np.array(neurons_this_depth[rois_min:rois_max + 1])

                                    if len(select_rois) > 1:
                                        pdf_name = save_folder/('roi' + str(select_rois[0]) + '-' + str(
                                            select_rois[-1]) + '.pdf')
                                    elif len(select_rois) == 1:
                                        pdf_name = save_folder/('roi' + str(select_rois) + '.pdf')
                                    with PdfPages(pdf_name) as pdf:

                                        for choose_roi in select_rois:

                                            roi = choose_roi
                                            trace_arr_noblank, _ = create_trace_arr_per_roi(roi, dffs_ast_all, depth_list, stim_dict_all,
                                                                                            mode='sort_by_depth',
                                                                                            protocol='fix_length',
                                                                                            blank_period=0,
                                                                                            frame_rate=frame_rate)
                                            Z = trace_arr_noblank.flatten()

                                            Z_ = Z[~np.isnan(Z)]
                                            X_ = X[~np.isnan(Z)]
                                            Y_ = Y[~np.isnan(Z)]
                                            log_X_ = np.log(X_ * 100)
                                            log_Y_ = np.log(np.degrees(Y_))
                                            popt = results[results.ROI==roi].values[0][3:-1]
                                            r_sq = results[results.ROI == roi]['r_sq'].values[0]
                                            Z_fit_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt)
                                            Z_fit = np.empty(Z.shape)
                                            Z_fit[:] = np.NaN
                                            Z_fit[~np.isnan(Z)] = Z_fit_
                                            gaussian_blob_plotting(roi=roi, speed_arr_noblank=speed_arr,
                                                                    speed_thr=speed_thr,
                                                                    of_arr_noblank=of_arr_noblank,
                                                                    trace_arr_noblank=trace_arr_noblank,
                                                                    popt=popt, Z_fit=Z_fit, X=X, Y=Y, r_sq=r_sq, Z_=Z_, Z_fit_=Z_fit_)

                                            pdf.savefig()
                                            plt.close()

                                            print('ROI' + str(choose_roi), flush=True)
            
            print('Finished plotting.', flush=True)
    
    
    
if __name__ == "__main__":
    defopt.run(main)


    