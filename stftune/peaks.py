import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from note import Note

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

def plot_peaks(Zxx, f, t, **kwargs):
    """Plots the peaks detected in a given STFT matrix

    Args:
        Zxx (np.ndarray): Complex STFT values.
        f (np.ndarray): The frequencies for each row of Zxx.
        t (np.ndarray): The times for each column of Zxx.
        title (str, optional): Title for the chart. Defaults to "Peaks".
        kwargs: keyword arguments for scipy.signal.find_peaks
    """
    # default to "Peaks"
    title = kwargs.pop("title", "Peaks")

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

def find_notes(Zxx, f, t, freq_thresh, note_gap_time, min_note_length, **kwargs):
    """Finds the notes in a signal based on its STFT matrix

    Args:
        Zxx (np.ndarray): Complex STFT values.
        f (np.ndarray): The frequencies for each row of Zxx.
        t (np.ndarray): The times for each column of Zxx.
        freq_thresh (float): The max difference in frequency for a sample to be considered part of a note
        note_gap_time (float): The max time since a note was last present for a sample to be part of it
        min_note_length (float): The minimum length for a note

    Returns:
        list(note.Note): The notes in the signal that fit the criteria
    """

    peak_data = [detect_peaks(Zxx[:, i], f, **kwargs) for i in range(Zxx.shape[1])]
    notes = []
    for (peak_amp, peak_freq), time in zip(peak_data, t):
        if (len(peak_freq) == 0):
            continue
        fundamental = peak_freq[0]
        updated_note = False
        # iterate in reversed order since recent notes are more likely to match
        for note in reversed(notes):
            # once updated_note becomes True, we won't update any other ones
            updated_note = updated_note or note.update(fundamental, time)
        if not updated_note:
            # didn't find a matching note, so create a new one
            new_note = Note(freq_thresh, note_gap_time)
            new_note.update(fundamental, time)
            notes.append(new_note)

    return [note for note in notes if note.length > min_note_length]    

def plot_notes(notes, title="Notes"):
    """Plots the provided Notes

    Args:
        notes (list(note.Note)): The notes to plot.
        title (str, optional): Title for the chart. Defaults to "Notes".
    """
    plt.figure()
    for note in notes:
        x = [note.start_time, note.end_time]
        y = [note.frequency, note.frequency]
        plt.plot(x, y, label=note)
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.show()