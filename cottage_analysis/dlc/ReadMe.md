# DLC

## Installation

On CAMP I tried to run their conda env files and it failed. Instead I did:

### Nemo update

The nodes have been updated for CUDA 11. Also, it seems better to have an environment with no GUI support for training/inference in `sbatch` and a separate envoriment for GUI stuff. (https://github.com/DeepLabCut/DeepLabCut/issues/1583)

New install:

```
ml cuDNN/8.1.1.33-CUDA-11.2.1
conda create -n dlc_nogui -c conda-forge python=3.8
conda activate dlc_nogui
```

With the `ml` I should have all I need, but tensorflow was throwing lots of library warnings, so I also install the conda version.

```
conda install -c conda-forge cudnn=8.1 cudatoolkit=11.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/dlc_nogui/lib/
```


Add the `conda` part of DLC environment.yml, before installing DLC per se

```
conda install pip ipython nb_conda jupyter ffmpeg
pip install deeplabcut
```

Run `test_tf_install.py` to check that the GPU is detected as expected. See `slurm_train_dlc.sh` for example slurm with 
`ml` and `export`.

### Camp version
Attempt #5

```
srun -p vis -n 1 --gres=gpu:1 -t 00:30:00 --pty bash
ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130 Anaconda3

conda create -n DLC python=3.7 tensorflow-gpu=1.15 cudatoolkit=10.0.130
conda activate DLC
```

At that stage I can import tensorflow in python. Let add the requirements of DLC

More dependencies

```
conda install -c conda-forge statsmodels filterpy matplotlib pandas scikit-learn pyyaml six tqdm click //
        filterpy ruamel.yaml opencv scikit-image pytables wxpython jupyter nb_conda Shapely pip imgaug
```

At that point I start to need a GUI

## Starting the GUI

```
srun -p vis -n 1 --gres=gpu:1 -t 08:30:00 --pty bash
ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130 Anaconda3 Tigervnc fluxbox FFmpeg
vncstart
```

Connect to tigervpn and 

```
fluxbox & 
cd ~/home/users/blota/code/DeepLabCut/
conda activate DLC
python -m deeplabcut
```
