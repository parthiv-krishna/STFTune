import numpy as np
import samplerate

def stretch(x, scale_factor, N, overlap, fs=44_100, window=None):
    """Stretches a signal by scale_factor using the phase vocoder algorithm

    Args:
        x (np.ndarray): The input signal.
        scale_factor (float): The factor by which to stretch (should be > 1).
        N (int): Chunk size for the phase vocoder algorithm.
        overlap (int): The number of samples of overlap between chunks.
        fs (int): The sampling frequency in Hz. Defaults to 44_100.
        window (np.ndarray, optional): Window to use, must be of length N. Defaults to None (np.hanning(N)).

    Returns:
        np.ndarray: The stretched signal.
    """

    if len(x) < N:
        print(f"Could not stretch signal of length {len(x)} with chunks of size {N}")
        return x

    if window is None:
        window = np.hanning(N) # hanning window

    if len(window) != N:
        print(f"Window length should be N!")
        return x

    hop_in = N - overlap
    hop_out = round(hop_in * scale_factor)
    dt_in = hop_in/fs

    num_chunks = ((len(x)-N)//(hop_in)) + 1
    start = 0
    end = start + N

    # Preallocate space for result
    result = np.zeros(num_chunks*hop_out + N)

    freq_vector = np.fft.fftfreq(N, d=1/fs)

    # First chunk analysis
    y = x[start:end]
    yw = y * window
    YW = np.fft.rfft(yw)
    phase_in = np.angle(YW)
    phase_out = np.copy(phase_in)

    # Unchanged first chunk synthesis
    result[start:end] = y

    # Handle the remaining chunks
    for chunk_idx in range(1, num_chunks):
        start += N - overlap
        end = start + N

        prev_phase_in = np.copy(phase_in)
        prev_phase_out = np.copy(phase_out)

        # Analysis
        y = x[start:end]
        yw = y * window
        YW = np.fft.rfft(yw)
        phase_in = np.angle(YW)

        # Match phase between chunks to reduce audio artifacts
        cycle_add = dt_in * freq_vector[:N//2+1] - (phase_in - prev_phase_in) / (2*np.pi)
        cycle_add = np.round(cycle_add)
        phase_freq = (phase_in - prev_phase_in + 2*np.pi*cycle_add) / (2*np.pi*dt_in)
        phase_out = prev_phase_out + 2*np.pi*phase_freq*dt_in
        YW_phased = np.abs(YW)*np.exp(1j*phase_out)

        # Synthesis
        synth = np.fft.irfft(YW_phased)
        start_out = chunk_idx*hop_out
        end_out = start_out + N
        result[start_out:end_out] += synth       

    # Normalize signal if it is outside [-1, 1]
    max_val = np.max(np.abs(result))
    if max_val > 1:
        result = result / max_val

    return result

def pitch_shift(x, freq_ratio, fs=44_100, N=2048, overlap=int(2048*0.75), window=None):
    """Pitch shifts a signal while attempting to avoid distortion/artifacts.

    Args:
        x (np.ndarray): The input signal.
        fs (int): The sampling frequency in Hz. Defaults to 44_100.
        freq_ratio (int): The ratio of out_freq/in_freq. E.g. 2 to increase by one octave, 2**(1/12) for one semitone.
        N (int, optional): Chunk size for the phase vocoder algorithm. Defaults to 2048.
        overlap (int, optional): The number of samples of overlap between chunks. Defaults to int(2048*0.75).
        window (np.ndarray, optional): Window to use, must be of length N. Defaults to None (np.hanning(N)).

    Returns:
        np.ndarray: The pitch shifted signal.
    """
    # Stretch the signal with phase vocoder
    stretched = stretch(x, fs, freq_ratio, N, overlap, window)
    # Resample to actually change the pitch
    resampled = samplerate.resample(np.array(stretched), 1/freq_ratio, "sinc_best")
    return resampled