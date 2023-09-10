import os

# os.environ['DLClight'] = 'True'
import sys
import matplotlib

matplotlib.use("Agg")  # make sure we use a backend that can run in headless mode

MODEL = "headfixed_track_eye"

if __name__ == "__main__":
    import socket

    hostname = socket.gethostname()
    print("Running on %s" % hostname)
    if hostname == "C02Z85AULVDC":
        # that's my laptop
        ROOT_DIR = "/Volumes/lab-znamenskiyp/home/"
    else:
        # should be on camp
        ROOT_DIR = "/camp/lab/znamenskiyp/home/"

    model_folder = f"shared/projects/DLC_models/{MODEL}"
    config_file = os.path.join(ROOT_DIR, model_folder, "config.yaml")

    import tensorflow

    print(f"Using tensorflow {tensorflow.__version__}", flush=True)
    from tensorflow.python.client import device_lib

    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print("Available devices:")
    print(get_available_devices(), flush=True)

    import deeplabcut

    print(f"TRAIN model {MODEL} using {config_file}")
    deeplabcut.train_network(config_file, maxiters=50000, saveiters=1000)
    print("Done")
