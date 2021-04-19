# DLC

## Installation

On CAMP I tried to run their conda env files and it failed. Instead I did:


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
conda install -c conda-forge statsmodels filterpy matplotlib pandas scikit-learn pyyaml six tqdm click filterpy ruamel.yaml opencv scikit-image pytables wxpython jupyter nb_conda Shapely pip
```
At that point I start to need a GUI


## Starting the GUI

```
srun -p vis -n 1 --gres=gpu:1 -t 00:30:00 --pty bash
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
