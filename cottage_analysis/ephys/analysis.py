from pathlib import Path
import numpy as np
from scipy import signal
import json
from tqdm import tqdm
from spikeinterface.core.loading import load as si_load
from spikeinterface import preprocessing as spre
from znamutils import slurm_it


def load_raw_rec_from_aind(preprocessed_json_file):
    """Temporary function to load raw recording from AIND preprocessed data"""
    preprocessed_json_file = Path(preprocessed_json_file)
    with open(preprocessed_json_file, "r") as f:
        preprocessing_vizualization_data = json.load(f)

    recording_full_dict = preprocessing_vizualization_data["recording"]["timeseries"][
        "full"
    ]["raw"]
    rec = si_load(recording_full_dict, base_folder=preprocessed_json_file.parent)
    return rec


@slurm_it(conda_env="onix-3dvision")
def compute_lfp_power_spectrum(
    recording_path, output_folder=None, cutoff=300, use_load_raw_rec_from_aind=False
):
    """
    Compute the average LFP power spectrum for a recording.

    Args:
        recording: SpikeInterface recording object or path to the recording folder.
        output_folder: Path to save the results (.npz file).
        cutoff: Low-pass filter cutoff frequency (Hz).

    Returns:
        power_spectrums: The computed average power spectrum.
    """
    if use_load_raw_rec_from_aind:
        recording = load_raw_rec_from_aind(recording_path)
    else:
        recording = si_load(recording_path)
    fs = cutoff * 3
    recording = spre.bandpass_filter(
        recording, freq_min=0.5, freq_max=cutoff, ignore_low_freq_error=True
    )
    recording = spre.resample(recording, fs)

    power_spectrums = None
    total_length = recording.get_total_duration()
    chunk_size = 60
    nchunks = int(total_length // chunk_size + 1)

    for ichunk in tqdm(range(nchunks), total=nchunks, desc="Computing LFP PSD"):
        trace = recording.get_traces(
            start_frame=int(ichunk * fs * chunk_size),
            end_frame=min(
                int((ichunk + 1) * fs * chunk_size), recording.get_num_frames()
            ),
            return_in_uV=True,
        )
        f, Pxx_den = signal.welch(trace.T, fs, nperseg=2048)
        if power_spectrums is None:
            power_spectrums = np.zeros(Pxx_den.shape)
        power_spectrums += Pxx_den / nchunks

    if output_folder is not None:
        target = Path(output_folder) / "lfp_power_spectrum.npz"
        np.savez(
            target,
            power_spectrums=power_spectrums,
            f=f,
            fs=fs,
            cutoff=cutoff,
            nchunks=nchunks,
        )
    return power_spectrums
