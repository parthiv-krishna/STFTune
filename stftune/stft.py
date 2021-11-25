import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def analysis(x, fs=44_100, N=256):
    """Performs STFT analysis of a given signal

    Args:
        x (np.ndarray): The signal to analyze.
        fs (int, optional): Sampling frequency in Hz. Defaults to 44_100.
        N (int, optional): Length of each STFT "chunk". Defaults to 256.

    Returns:
        Zxx: np.ndarray with the complex STFT values.
        f: np.ndarray with the frequencies for each row of Zxx.
        t: np.ndarray with the times for each column of Zxx.
    """

    f, t, Zxx = scipy.signal.stft(x, 
                                  fs=fs,
                                  window="hann",
                                  nperseg=N)

    return Zxx, f, t

def synthesis(Zxx, fs=44_100):
    """Synthesizes a signal from a given STFT matrix

    Args:
        Zxx (np.ndarray): Complex STFT values.
        fs (int, optional): Sampling frequency in Hz. Defaults to 44_100.

    Returns:
        x: np.ndarray containing the synthesized signal.
    """
    t, x = scipy.signal.istft(Zxx,
                                   fs=fs,
                                   window="hann")
    
    return x
