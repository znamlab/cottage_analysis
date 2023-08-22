#This is an example script to run the functions in camera_extrinsics.py, with all the definitions you'll need. 



from cottage_analysis.imaging.arena_cameras import camera_extrinsics as camex
from cottage_analysis.io_module import onix as onix
from pathlib import Path
import znamcalib.calibrate_lighthouse as light

DATA_PATH = Path('/camp/lab/znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote')
PROCESSED_PATH = Path('/camp/lab/znamenskiyp/home/shared/projects/blota_onix_pilote')
INTRINSICS_PATH = Path('/camp/lab/znamenskiyp/home/shared/projects/blota_onix_calibration/camera_intrinsics')
EXTRINSICS_PATH = Path('/camp/lab/znamenskiyp/home/shared/projects/blota_onix_calibration/arena_extrinsics')
INTRINSICS_SESSION = 'S20220627'
EXTRINSICS_SESSION = 'S20230509'

MOUSE = 'BRAC7448.2d'
SESSION = 'S20230412'
CAMERA = 'cam2_camera'

#######

#define function to find 'ephys' and 'camera'

ephys = 'R163257'
camera = 'R162624_freelymoving'

########

data_path = DATA_PATH / MOUSE / SESSION / ephys
camera_path = DATA_PATH / MOUSE / SESSION / camera
processed_path = PROCESSED_PATH / MOUSE / SESSION
intrinsics_path = INTRINSICS_PATH / INTRINSICS_SESSION / CAMERA
extrinsics_path = EXTRINSICS_PATH / EXTRINSICS_SESSION / CAMERA / 'cam2_camera_snapshot_0_extrinsics_0'
video_path = '/camp/lab/znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote/BRAC7448.2d/S20230412/R162624_freelymoving/cam2_camera_2023-04-12T16_26_24.mp4'
lighthouse_calibration = Path('/camp/lab/znamenskiyp/data/instruments/raw_data/projects/blota_onix_calibration/lighthouse_calibration/S20230412')


########Reading intrinsic and extrinsic calibrations.#########

# Choose 1 marker and extract the relevant arguments for OpenCV functions: rvec and tvec

extrinsics = camex.load_extrinsics(extrinsics_path)

rvec = extrinsics['marker4']['rvec']
tvec = extrinsics['marker4']['tvec']

# Choose 1 marker and extract the relevant arguments for OpenCV functions: cameraMatrix and distCoeffs

intrinsics = camex.load_intrinsics(intrinsics_path)

cameraMatrix = intrinsics['mtx']
distCoeffs = intrinsics['dist']

#######Reading Lighthouse data#####

photodiode = onix.load_ts4231(data_path)

#Choose a diode

diode3 = photodiode[3]

#Specify a Lighthouse calibration

calibration = Path('/camp/lab/znamenskiyp/data/instruments/raw_data/projects/blota_onix_calibration/lighthouse_calibration/S20230412')

#calibrate Lighthouse data

transform_matrix = light.calibrate_session(lighthouse_calibration, 20)
trans_data = light.transform_data(diode3, transform_matrix)

#######Loading camera metadata#####

camera_metadata = onix.load_camera_times(camera_path)
#converting camera timestamps into onix timestamps (old way)

camera_id = 'cam2_camera_2023-04-12T16_26_24' #This example is from before we forced the camera timestamp csvs 
#to not include the timestamp in the name

cam2_metadata = camex.dio_conversion(camera_id, camera_metadata, data_path)

#plotting a static time/orientation

onix_time = 240000

camex.plot_onix_time(onix_time, video_path, trans_data, cam_metadata=cam2_metadata, 
                     rvec = rvec, tvec = tvec, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)




