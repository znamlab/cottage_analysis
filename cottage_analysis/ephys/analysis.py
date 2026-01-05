from pathlib import Path
import numpy as np
from scipy import signal
from tqdm import tqdm
from spikeinterface.core.loading import load as si_load
from spikeinterface import preprocessing as spre


def compute_lfp_power_spectrum(recording_folder, output_folder=None, cutoff=300):
    """
    Compute the average LFP power spectrum for a recording.

    Args:
        recording_folder: Path to the SpikeInterface recording folder.
        output_folder: Path to save the results (.npz file).
        cutoff: Low-pass filter cutoff frequency (Hz).

    Returns:
        power_spectrums: The computed average power spectrum.
    """
    recording = si_load(recording_folder)
    fs = cutoff * 3
    recording = spre.bandpass_filter(recording, freq_min=0.1, freq_max=cutoff)
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
