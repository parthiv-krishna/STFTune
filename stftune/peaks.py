import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def detect_peaks(Zxx, f, **kwargs):
    """Detects peaks in a column of an STFT spectrogram

    Args:
        Zxx (np.ndarray): Complex STFT values.
        f (np.ndarray): The frequencies for each row of Zxx.
        kwargs: keyword arguments for scipy.signal.find_peaks

    Returns:
        peak_amp: np.ndarray with the amplitude values for the peaks
        peak_freq: np.ndarray with the frequency values for the peaks

    """

    Sxx = np.abs(Zxx)
    
    peaks, properties = scipy.signal.find_peaks(Sxx, **kwargs)
    
    num_peaks = len(peaks)
    
    peak_amp = np.zeros(num_peaks)
    peak_freq = np.zeros(num_peaks)
    for i, peak_idx in enumerate(peaks):
        peak_Sxx = Sxx[peak_idx-1:peak_idx+2] # actually i-1, i, i+1
        peak_f = f[peak_idx-1:peak_idx+2] # actually i-1, i, i+1
        peak_amp[i] = np.average(peak_Sxx)
        peak_freq[i] = np.average(peak_f, weights=peak_Sxx) # weighted average
        
    return peak_amp, peak_freq

def plot_peaks(Zxx, f, t, title="Peaks", **kwargs):
    """Plots the peaks detected in a given STFT matrix

    Args:
        Zxx (np.ndarray): Complex STFT values.
        f (np.ndarray): The frequencies for each row of Zxx.
        t (np.ndarray): The times for each column of Zxx.
        title (str, optional): Title for the chart. Defaults to "Peaks".
        kwargs: keyword arguments for scipy.signal.find_peaks
    """
    peak_data = [detect_peaks(Zxx[:, i], f, **kwargs) for i in range(Zxx.shape[1])]
    plt.figure()
    for (peak_amp, peak_freq), time in zip(peak_data, t):
        # array of the same time value to allow scatter to display properly
        const_x = np.ones_like(peak_freq) * time
        plt.scatter(const_x, peak_freq, s=1, c='b')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()
