import numpy as np

def generate_sinusoid(f, secs=1, fs=44_100, return_t=False):
    """Generates a sinusoid at the given frequency

    Args:
        f (int | float): The desired frequency in Hz.
        secs (int | float, optional): The desired length (in seconds). Defaults to 1.
        fs (int, optional): Sampling frequency in Hz. Defaults to 44_100.
        return_t (bool, optional): Whether to return the time array. Defaults to False.

    Returns:
        x: np.ndarray containing the sinusoidal signal.
        t: np.ndarray containing the corresponding time values. Returned only if return_t is True.
    """
    t = np.arange(fs*secs)/fs
    x = np.sin(2*np.pi*f*t)
    
    if return_t:
        x, t
    return x


def generate_multi_sinusoid(f_arr, secs_arr, fs=44_100, return_t=False):
    """Generates a signal composed of multiple sequential sinusoids

    Args:
        f_arr (iterable(int | float)): Array of frequencies in Hz.
        secs_arr (iterable(int | float)): Array of lengths in seconds corresponding to each frequency.
        fs ([type], optional): [description]. Defaults to 44_100.
        return_t (bool, optional): Whether to return the time array. Defaults to False.

    Returns:
        x: np.ndarray containing the sinusoidal signal.
        t: np.ndarray containing the corresponding time values. Returned only if return_t is True.
    """
    sinusoids = []
    total_secs = 0
    for f, secs in zip(f_arr, secs_arr):
        sinusoids.append(generate_sinusoid(f, secs, fs=fs))
        total_secs += secs

    x = np.concatenate(sinusoids)
    if return_t:
        t = np.arange(fs*total_secs) / fs
        return x, t
    return x
