"""This script is intended to be copied and edited by find_pupil.fit_ellipses"""

from pathlib import Path
from cottage_analysis.eye_tracking import eye_model_fitting

dlc_file = "XXX_DLC_FILE_XXX"
target_folder = "XXX_TARGET_XXX"
likelihood = "XXX_LIKELIHOOD_XXX"
label = "XXX_LABEL_XXX"
origin_id = "XXX_ORIGIN_ID_XXX"
project = "XXX_PROJECT_XXX"

if origin_id.startswith("XXX_"):
    origin_id = None
if likelihood.startswith("XXX_"):
    likelihood = None


dlc_file = Path(dlc_file)
assert dlc_file.exists()

target = Path(target_folder) / f"{dlc_file.stem}_ellipse_fits.csv"
print(f"Doing %s" % dlc_file)
ellipse_fits = eye_model_fitting.fit_ellipses(dlc_file, likelihood_threshold=likelihood)
ellipse_fits.to_csv(target, index=False)


if __name__ == "__main___":
    if plot_fits:
        raw_path = Path(PARAMETERS["data_root"]["processed"])
        target_file = cam_path / "eye_tracking_ellipse_not_filtered.mp4"
        raw_data = raw_path / project / mouse / session / recording / camera
        video_file = raw_data / "{0}.mp4".format(model[: model.find("DLC")])

        assert target_file.parent.is_dir()
        REDO = False
        if not target_file.exists() or REDO:
            ellipse = EllipseModel()

            dlc_res = dlc_results[model]
            ellipse_fits = []

            fig = plt.figure()
            fig.set_size_inches((9, 3))

            img = get_img_from_fig(fig)
            cam_data = cv2.VideoCapture(str(video_file))
            fps = cam_data.get(cv2.CAP_PROP_FPS)
            fcc = int(cam_data.get(cv2.CAP_PROP_FOURCC))
            fcc = (
                chr(fcc & 0xFF)
                + chr((fcc >> 8) & 0xFF)
                + chr((fcc >> 16) & 0xFF)
                + chr((fcc >> 24) & 0xFF)
            )

            output = cv2.VideoWriter(
                str(target_file),
                cv2.VideoWriter_fourcc(*fcc),
                fps / 4,
                (img.shape[1], img.shape[0]),
            )

            for frame_id, track in dlc_res.iterrows():
                # plot
                fig.clear()
                ax_img = fig.add_subplot(1, 3, 1)
                ax_track = fig.add_subplot(1, 3, 2)
                ax_fit = fig.add_subplot(1, 3, 3)
                fig.subplots_adjust(
                    left=0, right=1, bottom=0, top=1, wspace=0, hspace=0
                )

                ret, frame = cam_data.read()

                for ax in [ax_img, ax_fit, ax_track]:
                    ax.imshow(frame)
                    ax.set_yticks([])
                    ax.set_xticks([])

                ax_track.scatter(xdata, ydata, s=likelihood * 10)
                circ_coord = ellipse.predict_xy(np.arange(0, 2 * np.pi, 0.1))
                ax_fit.plot(circ_coord[:, 0], circ_coord[:, 1])
                write_fig_to_video(fig, output)
            cam_data.release()
            output.release()
