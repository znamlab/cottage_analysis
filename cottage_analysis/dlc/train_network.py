import os
# os.environ['DLClight'] = 'True'
import sys
import matplotlib
matplotlib.use('Agg')  # make sure we use a backend that can run in headless mode


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    print('Running on %s' % hostname)
    if hostname == 'C02Z85AULVDC':
        # that's my laptop
        ROOT_DIR = "/Volumes/lab-znamenskiyp/home/"
    else:
        # should be on camp
        ROOT_DIR = "/camp/lab/znamenskiyp/home/"

    model_folder = "shared/projects/DLC_models/all_eyes_2023"
    config_file = os.path.join(ROOT_DIR, model_folder, "config.yaml")

    import deeplabcut
    print("TRAIN")
    deeplabcut.train_network(config_file)
    print("Done")
