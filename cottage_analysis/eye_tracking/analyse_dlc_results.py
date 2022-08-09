"""
Temporary script to analyse the results out of DLC
"""



ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/"
DATA_DIR = "3d_vision/EyeCamCalibration/RightEyeCam/TrainingData/twophoton/"

VIDEO_NAME = 'PZAH4.1c_S20210407_R174323_right_eye_camera_sample'
MODEL = 'DLC_resnet_50'

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from skimage.measure import EllipseModel

    main_folder = os.path.join(ROOT_DIR, DATA_DIR)
    output_folder = '/Users/blota/OneDrive - The Francis Crick Institute/presentations/20210505_labmeeting/'
    fnames = [f for f in os.listdir(main_folder) if f.startswith(VIDEO_NAME + MODEL)]
    h5file = [f for f in fnames if f.endswith('h5')]
    assert len(h5file) == 1
    h5file = h5file[0]
    df = pd.read_hdf(os.path.join(ROOT_DIR, DATA_DIR, h5file))
    # I have only one scorer for now
    df = df['DLC_resnet_50_right_eye_cam_twopApr22shuffle1_250000']
    print(df.head())

    # fit an ellipse
    param_order = ['xc', 'yc', 'a', 'b', 'theta']
    for w in param_order:
        df[w] = np.nan
    pupil = np.arange(1, 13)
    pupil_str = ['pupil_%d' % i for i in pupil]
    for index, series in df.iterrows():
        ellipse = EllipseModel()
        xy = series.loc[pupil_str, ['x', 'y']].values.reshape(len(pupil), 2)
        success = ellipse.estimate(xy)
        if not success:
            continue
        for k, v in zip(param_order, ellipse.params):
            df.loc[index, k] = v

    # get the video
    video_file = os.path.join(main_folder, VIDEO_NAME + '.avi')
    assert os.path.isfile(video_file)
    video = cv2.VideoCapture(os.path.join(main_folder, video_file))
    frame = 0
    fig = plt.figure()
    for n in range(2, 1000, 100):

        while frame < n:
            video.read()
            frame += 1
        ret, first_frame = video.read()
        frame += 1
        fig.clear()

        ax = fig.add_subplot(111)
        ax.imshow(first_frame)
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        ax.set_xlim([850, 1200])
        ax.set_ylim([350, 0])
        fig.savefig(os.path.join(output_folder, 'twop_image_example_frame%d.png' % n), dpi=600)

        series = df.loc[n + 2]
        markers = ['reflection'] + pupil_str + ['eye_%d' % i for i in [12, 3, 6, 9]]
        x = series.loc[markers, ['x']]
        y = series.loc[markers, ['y']]
        ax.scatter(x, y, c=np.arange(len(x)))
        fig.savefig(os.path.join(output_folder, 'twop_image_example_frame%d_labeled.png' % n ), dpi=600)
    video.release()

    # likelihood filter
    fig.clear()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.imshow(first_frame)
    sc = ax.scatter(df.reflection.x, df.reflection.y, c=df.reflection.likelihood, s=2)
    cb0 = plt.colorbar(sc)
    cb0.set_label('Likelihood')
    ax.set_title('Eye center')
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95, right=0.95)
    fig.savefig(os.path.join(output_folder, 'likelihood_overview.png'), dpi=300)

    ax.set_xlim([850, 1400])
    ax.set_ylim([350, 0])
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95, right=0.95)
    fig.savefig(os.path.join(output_folder, 'likelihood_eye.png'), dpi=300)

    fig.clear()
    threshold = 0.95
    ax = fig.add_subplot(2, 2, 1)
    ax.clear()
    ax.hist(df.reflection.likelihood, bins=np.arange(0, 1.01, 0.01),
            cumulative=True, density=True, log=True)
    ax.set_xlim(0, 1)
    ax.axvline(threshold, color='r')
    ax.set_xlabel('Likelihood')
    ax.set_title('Eye center')
    ax.set_ylabel('CDF')

    ax = fig.add_subplot(2, 2, 2)
    ax.clear()
    all_like = df.xs('likelihood', axis=1, level=1).values
    ax.hist(all_like.flatten(), bins=np.arange(0, 1.01, 0.01),
            cumulative=True, density=True, log=True)
    ax.set_xlim(0, 1)
    ax.axvline(threshold, color='r')
    ax.set_xlabel('Likelihood')
    ax.set_title('All markers')
    ax.set_ylabel('CDF')

    ax = fig.add_subplot(2, 2, 3)
    ok = df.reflection.likelihood > threshold
    ax.imshow(first_frame)
    sc = ax.scatter(df[ok].reflection.x, df[ok].reflection.y, c=df[ok].index/60, s=2)
    cb0 = plt.colorbar(sc)
    cb0.set_label('Time (s)')
    ax.set_title('Eye center')
    ax.set_xlim([850, 1200])
    ax.set_ylim([350, 0])

    ax = fig.add_subplot(2, 2, 4)
    ax.clear()
    ax.plot(df[ok].reflection.index/60, df[ok].reflection.x - np.median(df.reflection.x), label='X position')
    ax.plot(df[ok].reflection.index / 60, df[ok].reflection.y - np.median(df.reflection.y), label='Y position')
    ax.legend(loc=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (relative to median)')
    ax.set_xlim([0, df.index.max()/60])

    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95, right=0.95)
    fig.savefig(os.path.join(output_folder, 'likelihood_threshold.png'), dpi=300)

    fig.clear()
    ax = fig.add_subplot(2, 2, 1, aspect='equal')
    ax.clear()
    ax.imshow(first_frame)
    cmap = plt.get_cmap('viridis', 4)
    color = {n: cmap(i) for i, n in enumerate(np.arange(3, 13, 3))}
    for i in [12, 9, 3, 6]:
        d = df['eye_%d' % i]
        d = d[d.likelihood > threshold]
        sc2 = ax.plot(d.x, d.y, color=color[i], ms=2, marker='o', linestyle='')
    ax.set_xlim([850, 1200])
    ax.set_ylim([350, 0])
    centers = dict()
    for i, eye in enumerate(np.arange(3, 13, 3)):
        d = df['eye_%d' % eye]
        d = d[d.likelihood > threshold]
        centers[eye] = np.array([np.median(d.x), np.median(d.y)])
        ax.plot(centers[eye][0], centers[eye][1], '+', color='darkred')

    ax = fig.add_subplot(2, 2, 2, aspect='equal')
    ax.clear()
    ax.imshow(first_frame)
    cmap = plt.get_cmap('viridis', 4)
    color = {n: cmap(i) for i, n in enumerate(np.arange(3, 13, 3))}
    okdf = df[df.eye_12.y > 100]
    for i in [12, 9, 3, 6]:
        d = okdf['eye_%d' % i]
        d = d[d.likelihood > threshold]
        sc2 = ax.plot(d.x, d.y, color=color[i], ms=2, marker='o', linestyle='')
    ax.set_xlim([850, 1200])
    ax.set_ylim([350, 0])
    ax.axhline(100, color=cmap(4))

    ax = fig.add_subplot(2, 2, 3)
    ax.clear()
    for i, eye in enumerate(np.arange(3, 13, 3)):
        d = df['eye_%d' % eye]
        d = d[d.likelihood > threshold]
        d2c = np.sqrt((d.x - centers[eye][0])**2 + (d.y - centers[eye][1])**2)
        ax.plot(d.index / 60, d2c, color=cmap(i))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to center')
    ax.set_xlim([0, df.index.max() / 60])

    ax = fig.add_subplot(2, 2, 4)
    ax.clear()
    okdf = df[(df.eye_12.likelihood > threshold) & (df.eye_6.likelihood > threshold)]
    t2b = np.sqrt((okdf.eye_12.x - okdf.eye_6.x)**2 + (okdf.eye_12.y - okdf.eye_6.y)**2)
    ax.plot(okdf.index / 60, t2b)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Eye opening')
    ax.set_xlim([0, df.index.max()/60])
    fig.savefig(os.path.join(output_folder, 'eye_opening.png'), dpi=600)

    fig.clear()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.clear()
    ax.imshow(first_frame)
    ax.set_xlim([850, 1200])
    ax.set_ylim([350, 0])
    frame_index = 102
    series = df.loc[frame_index]
    cmap = plt.get_cmap('viridis', 12)
    ax.scatter(series.loc[pupil_str, 'x'], series.loc[pupil_str, 'y'], c=[cmap(i) for i in range(12)],
               zorder=10)

    ellipse = EllipseModel()
    ellipse.params = series[param_order].values
    xy = ellipse.predict_xy(np.linspace(0, 2*np.pi, 100))
    ax.plot(xy[:, 0], xy[:, 1], 'k')
    ax.plot(series['xc'], series['yc'], 'o', color='k')
    # the ellipse is defined by:
    # xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
    # yt  = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
    xy = np.zeros([4, 2])
    for i, t in enumerate(np.arange(0, 2*np.pi, np.pi/2)):
        xy[i] = ellipse.predict_xy(t)
    ax.plot(xy[[0, 2], 0], xy[[0, 2], 1], color='k')
    ax.plot(xy[[1, 3], 0], xy[[1, 3], 1], color='k')
    fig.savefig(os.path.join(output_folder, 'ellipse_fit.png'), dpi=600)

    fig.clear()
    ax = fig.add_subplot(2, 2, 1)
    ax.clear()
    okdf = df[df.reflection.likelihood > threshold]
    ax.plot(okdf.index/60, okdf.xc - np.median(okdf.xc), label='X position')
    ax.plot(okdf.index/60, okdf.yc - np.median(okdf.yc), label='Y position')
    ax.set_xlim([0, df.index.max()/60])
    ax.legend(loc=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from median (px)')

    ax = fig.add_subplot(2, 2, 2)
    ax.clear()
    ax.plot(okdf.index/60, okdf.xc - np.median(okdf.xc), label='_nolegend_')
    ax.plot(okdf.index/60, okdf.yc - np.median(okdf.yc), label='_nolegend_')
    dst = np.sqrt((okdf.xc - np.median(okdf.xc)) ** 2 + (okdf.yc - np.median(okdf.yc)) ** 2)
    ax.plot(okdf.index[:-1] / 60, np.diff(dst),
            label='Instantaneous mvt', color='k')
    ax.legend(loc=0)
    ax.axhline(-3, color='darkred')
    ax.axhline(3, color='darkred')
    ax.set_xlim(np.array([0, 30]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from median (px)')

    ax = fig.add_subplot(2, 2, 3)
    ax.clear()
    imvt = np.abs(np.diff(dst))
    ax.hist(imvt, bins=np.arange(imvt.min(), imvt.max(), 0.1),
            log=True, density=True)
    ax.axhline(1/100, color='k')
    ax.axvline(3, color='darkred')
    ax.set_xlim([0, imvt.max()])
    ax.set_xlabel('Abs instantaneous mvt (px)')
    ax.set_ylabel('PDF')

    ax = fig.add_subplot(2, 2, 4)
    ax.clear()
    t_saccade = okdf.index[:-1][np.diff(dst) > 3]/60
    isi = np.diff(t_saccade)
    isi = isi[isi > 1/59]
    ax.hist(isi, bins=np.arange(0, isi.max(), 0.5))
    ax.set_xlabel('Inter saccade interval (s)')
    ax.set_ylabel('Count')
    ax.set_xlim([0, isi.max()])
    ax.text(0.1, 0.7, 'Average rate:\n      %.2f Hz' % (1/isi.mean()), transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='left')
    fig.savefig(os.path.join(output_folder, 'saccades.png'), dpi=300)

    fig.clear()
    ax = fig.add_subplot(2, 2, 1, aspect='equal')
    ax.clear()
    ax.imshow(first_frame)
    ax.set_xlim([850, 1200])
    ax.set_ylim([350, 0])
    sc = ax.scatter(okdf.xc, okdf.yc, c=okdf.index/60, s=1)
    cb = plt.colorbar(sc)

    ax = fig.add_subplot(2, 2, 2, aspect='equal')
    ax.clear()
    im = ax.hexbin(okdf.xc, okdf.yc, gridsize=100, vmax=50)
    ax.set_ylim([190, 160])
    ax.set_xlim([1010, 1050])
    cb = plt.colorbar(im)
    cb.set_label('# of frames')

    from sklearn import mixture

    clf = mixture.GaussianMixture(n_components=4, covariance_type='full')
    clf.fit(okdf[['xc', 'yc']].values)
    pred = clf.predict(okdf[['xc', 'yc']].values)

    ax = fig.add_subplot(2, 2, 3, aspect='equal')
    ax.clear()
    sc = ax.scatter(okdf.xc, okdf.yc, c=pred, s=2)
    cb = plt.colorbar(sc)
    cb.set_label('Kluster')
    ax.set_xlim([980, 1070])
    ax.set_ylim([220, 150])

    ax = fig.add_subplot(2, 2, 4)
    ax.clear()
    pos_change_time = okdf[:-1].index[np.diff(pred) != 0] / 60
    pos_change_inter_time = np.diff(pos_change_time)
    ax.hist(pos_change_inter_time, density=True,
            histtype='step', cumulative=False, log=True, bins=np.arange(0, pos_change_inter_time.max(), 0.1))
    ax.set_xlabel('Inter-cluster transition time')
    ax.set_ylabel('PDF')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, pos_change_inter_time.max()])

    fig.savefig(os.path.join(output_folder, 'eye_position_clustering.png'), dpi=300)

    fig = plt.figure()
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ellipse = EllipseModel()
    ax0.clear()
    ax1.clear()
    for index, series in okdf[::100].iterrows():
        ellipse.params = series[param_order].values
        xy = np.zeros([4, 2])
        for i, t in enumerate(np.arange(0, 2 * np.pi, np.pi / 2)):
            xy[i] = ellipse.predict_xy(t)
        if series.a.values[0] > series.b.values[0]:
            axes = [ax0, ax1]
        else:
            axes = [ax1, ax0]
        for i, p in enumerate([(0, 2), (1, 3)]):
            x, y = xy[[p[0], p[1]], 0], xy[[p[0], p[1]], 1]
            slope = np.diff(y)/np.diff(x)
            b = y[0] - slope * x[0]
            x = np.hstack([900, x, 1200])
            y = np.hstack([slope*900+b, y, slope*1200+b, ])
            axes[i].plot(x, y, '-', color='darkred', alpha=0.1)
        xy = ellipse.predict_xy(np.linspace(0, 2 * np.pi, 100))
        ax2.plot(xy[:, 0], xy[:, 1], 'darkred', alpha=0.1)
        for ax in [ax0, ax1, ax2]:
            ax.imshow(first_frame)
            ax.set_ylim([250, 100])
            ax.set_xlim([950, 1100])

    print('Done')
