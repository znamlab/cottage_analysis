# DLC

## Installation

On CAMP I tried to run their conda env files and it failed. Instead I did:

```
srun -p vis -n 1 --gres=gpu:1 -t 00:30:00 --pty --x11 bash
ml TensorFlow/1.12.0-foss-2018b-Python-3.6.6
ml Anaconda3
```

Attempt #5

```
srun -p vis -n 1 --gres=gpu:1 -t 00:30:00 --pty --x11 bash
ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130
ml Anaconda3

conda create -n DLC python=3.7 tensorflow-gpu=1.15 cudatoolkit=10.0.130
conda activate DLC
```

At that stage I can import tensorflow in python. Let add the requirements of DLC

More dependencies

```
conda install matplotlib pandas scikit-learn
conda install pyyaml six tables tqdm click filterpy
conda install ruamel.yaml opencv scikit-image
```
At that point I start to need a GUI

```
ml Tigervnc
vncstart
ml fluxbox 
fluxbox
```

It goes a bit further. New dependencies

```
conda install -c conda-forge statsmodels filterpy
```

Seems to work.

## Starting the GUI

We need to start anaconda and load the relevant cuda/cudnn for DLC but if we want the GUI we also need Tigervnc and fluxbox. See the script `start_gui_on_camp.sh`
