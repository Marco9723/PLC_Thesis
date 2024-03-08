import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy
sys.path.append("/nas/home/mviviani/nas/home/mviviani/tesi")
#from CODE.config import CONFIG    #  + CANCELLARE FUNZIONI MAI UTILIZZATE !!!
from config import CONFIG

def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def visualize(target, input, recon, path):
    sr = CONFIG.DATA.sr
    window_size = 1024
    window = np.hanning(window_size)

    stft_hr = librosa.core.spectrum.stft(target, n_fft=window_size, hop_length=512, window=window)
    stft_hr = 2 * np.abs(stft_hr) / np.sum(window)

    stft_lr = librosa.core.spectrum.stft(input, n_fft=window_size, hop_length=512, window=window)
    stft_lr = 2 * np.abs(stft_lr) / np.sum(window)

    stft_recon = librosa.core.spectrum.stft(recon, n_fft=window_size, hop_length=512, window=window)
    stft_recon = 2 * np.abs(stft_recon) / np.sum(window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 10))
    ax1.title.set_text('Target signal')
    ax2.title.set_text('Lossy signal')
    ax3.title.set_text('Reconstructed signal')

    canvas = FigureCanvas(fig)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_hr), ax=ax1, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_lr), ax=ax2, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_recon), ax=ax3, y_axis='linear', x_axis='time', sr=sr)
    mkdir_p(path)
    fig.savefig(os.path.join(path, 'spec.png'))


def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=960)
    S2 = get_power(x_pr, nfft=960)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high

def calculate_sdr(original_signal, processed_signal):
    if len(original_signal) != len(processed_signal):
        raise ValueError("Signals have different lengths")
    #sdr = 10 * np.log10(numerator + 1e-6) - 10 * np.log10(denominator + 1e-6)

    numerator = np.sum(np.square(original_signal))
    denominator = np.sum(np.square(original_signal - processed_signal))
    
    sdr = 10 * np.log10(numerator / denominator)
    return sdr


def simulate_packet_loss(y_ref: np.ndarray, trace: np.ndarray, packet_dim: int) -> np.ndarray:
    # Copy the clean signal to create the lossy signal
    y_lost = deepcopy(y_ref)

    # Simulate packet losses according to given trace
    for i, loss in enumerate(trace):
        if loss:
            idx = i * packet_dim
            y_lost[idx: idx + packet_dim] = 0.

    return y_lost

def similarity_index(original_signal, reconstructed_signal):
    numerator = 2 * np.sum(original_signal * reconstructed_signal)
    denominator = np.sum(original_signal**2) + np.sum(reconstructed_signal**2)
    si = numerator / denominator
    return si

def relative_error(original_signal, reconstructed_signal):
    abs_difference = np.abs(original_signal - reconstructed_signal)
    relative_error_percentage = (np.sum(abs_difference) / np.sum(np.abs(original_signal))) * 100
    return relative_error_percentage