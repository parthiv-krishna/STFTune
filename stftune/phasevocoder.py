import numpy as np
import samplerate

def stretch(x, fs, scale_factor, N, overlap, window=None):
    if len(x) < N:
        print(f"Could not stretch signal of length {len(x)} with chunks of size {N}")
        return x

    if window is None:
        window = np.hanning(N) # hanning window

    hop_in = N - overlap
    hop_out = round(hop_in * scale_factor)
    deltaT_in = hop_in/fs

    num_chunks = ((len(x)-N)//(hop_in)) + 1
    start = 0
    end = start + N

    result = np.zeros(num_chunks*hop_out + N)

    freq_vector = np.fft.fftfreq(N, d=1/fs)

    # First chunk analysis
    y = x[start:end]
    yw = y * window
    YW = np.fft.rfft(yw)
    phase_in = np.angle(YW)
    phase_out = np.copy(phase_in)
    # Unchanged first frame synthesis
    result[start:end] = y

    for chunk_idx in range(1, num_chunks):
        start += N - overlap
        end = start + N

        prev_phase_in = np.copy(phase_in)
        prev_phase_out = np.copy(phase_out)

        # analysis
        y = x[start:end]
        yw = y * window
        YW = np.fft.rfft(yw)
        phase_in = np.angle(YW)

        num_cycles_to_true_freq = deltaT_in * freq_vector[:N//2+1] - (phase_in - prev_phase_in)/(2*np.pi)
        num_cycles_to_true_freq = np.round(num_cycles_to_true_freq)
        true_freq = (phase_in - prev_phase_in + 2*np.pi*num_cycles_to_true_freq) / (2*np.pi*deltaT_in)
        phase_out = prev_phase_out + 2*np.pi*true_freq*deltaT_in
        YW_phased = np.abs(YW)*np.exp(1j*phase_out)

        # Synthesis
        synth = np.fft.irfft(YW_phased)
        start_out = chunk_idx*hop_out
        end_out = start_out + N
        result[start_out:end_out] = synth       

    max_val = np.max(np.abs(result))
    if max_val > 1:
        result = result / max_val

    return result

def pitch_shift(x, fs, scale_factor, N=2048, overlap=int(2048*0.75), window=None, forceConstantLength=False):
    stretched = stretch(x, fs, scale_factor, N, overlap, window)
    resampled = samplerate.resample(np.array(stretched), 1/scale_factor, "sinc_best")
    print(type(stretched))
    if forceConstantLength:
        return stretch(resampled, fs, len(x)/len(resampled), N, overlap, window)
    return resampled