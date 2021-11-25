import numpy as np

class Note:
    """Class to store a note in the STFT spectrum.
    
    Each Note must be initialized/updated with the update method.
    Once all updates are complete, the start_time, end_time, length,
    and frequency properties will be populated and accessible.
    
    """

    def __init__(self, freq_thresh, note_gap_time):
        """Initializes an instance of the Note class

        Args:
            freq_thresh (int | float): Max difference between Note frequency and a new sample
            note_gap_time (float): Max gap between consecutive samples
        """
        self.freq = None
        self.sample_freqs = np.array([])
        self.sample_times = np.array([])

        self.freq_thresh = freq_thresh
        self.note_gap_time = note_gap_time

    def __str__(self):
        """Gives a string representation of the Note"""
        if self.freq is None:
            return "Uninitialized note"
        return f"{self.frequency:.2f} Hz from {self.start_time:.2f}s to {self.end_time:.2f}s"

    def __repr__(self):
        """Gives a string representation of the Note"""
        return self.__str__()

    def update(self, sample_freq, sample_time):
        """Updates the Note with a new sample

        Args:
            sample_freq (float): Frequency of the new sample
            sample_time (float): Time of the new sample

        Returns:
            True if the new sample was added to the note, False otherwise
        """

        should_update = False

        if self.freq is None:
            should_update = True
        else:
            freq_diff = abs(self.freq - sample_freq)
            time_diff = sample_time - self.sample_times[-1]
            if (freq_diff < self.freq_thresh and time_diff< self.note_gap_time):
                should_update = True
        
        if should_update:        
            self.sample_freqs = np.append(self.sample_freqs, sample_freq)
            self.sample_times = np.append(self.sample_times, sample_time)
            self.freq = np.mean(self.sample_freqs)

        return should_update

    @property
    def start_time(self):
        """Gets the start time of the Note, or None if uninitialized"""
        if self.freq is None:
            return None
        return self.sample_times[0]

    @property
    def end_time(self):
        """Gets the end time of the Note, or None if uninitialized"""
        if self.freq is None:
            return None
        return self.sample_times[-1]

    @property
    def length(self):
        """Gets the length of the Note, or None if uninitialized"""
        if self.freq is None:
            return 0
        return self.end_time - self.start_time
    
    @property
    def frequency(self):
        """Gets the frequency of the Note, or None if uninitialized"""
        return self.freq