import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Applies a low-pass Butterworth filter to the data.

    Parameters:
    data (array-like): The input signal.
    cutoff (float): The cutoff frequency of the filter.
    fs (int): The sampling rate of the signal.
    order (int): The order of the filter.

    Returns:
    array-like: The filtered signal.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def normalize_signal(data):
    """
    Normalizes the signal data to a range of [0, 1].

    Parameters:
    data (array-like): The input signal.

    Returns:
    array-like: The normalized signal.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def segment_signal(data, segment_length):
    """
    Segments the signal into smaller parts of equal length.

    Parameters:
    data (array-like): The input signal.
    segment_length (int): The number of samples in each segment.

    Returns:
    list: A list of segments.
    """
    return [data[i:i + segment_length] for i in range(0, len(data), segment_length) if i + segment_length <= len(data)]

def check_amplitude_variation(signal, threshold=0.5):
    """
    Checks if the amplitude variation of the signal exceeds a threshold.

    Parameters:
    signal (array-like): The PPG signal.
    threshold (float): The threshold for amplitude variation.

    Returns:
    bool: True if signal quality is good, False otherwise.
    """
    amplitude_variation = np.std(signal)
    return amplitude_variation < threshold


# # Example usage
# fs = 100  # Sampling frequency
# cutoff = 2.5  # Cutoff frequency for low-pass filter
# order = 5  # Order of the filter
# segment_length = 300  # Length of each segment (e.g., 3 seconds if fs is 100 Hz)

# # Apply preprocessing steps to each PPG signal
# for index, signal in enumerate(ppg_signals):
#     filtered_signal = butter_lowpass_filter(signal, cutoff, fs, order)
#     normalized_signal = normalize_signal(filtered_signal)
#     segments = segment_signal(normalized_signal, segment_length)

#     for segment in segments:
#         if assess_signal_quality(segment):
#             # Process good quality segments
#             pass

def frequency_analysis(signal, fs):
    """
    Performs frequency analysis on a PPG signal to identify the dominant frequency.

    Parameters:
    signal (array-like): The PPG signal.
    fs (int): The sampling frequency of the signal.

    Returns:
    float: The dominant frequency in the signal.
    """
    # Compute the Fast Fourier Transform (FFT)
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    fft_signal = fft(signal)
    
    # Find the absolute value of the FFT, which represents the power
    fft_power = np.abs(fft_signal)

    # Identify the dominant frequency
    dominant_frequency = freq[np.argmax(fft_power)]
    return dominant_frequency


def calculate_snr(signal, noise):
    """
    Calculates the Signal-to-Noise Ratio (SNR) of a PPG signal.

    Parameters:
    signal (array-like): The PPG signal.
    noise (array-like): The noise in the PPG signal.

    Returns:
    float: The SNR of the signal.
    """
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    
    # Ensure noise power is not zero to avoid division by zero
    if noise_power == 0:
        return np.inf
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr



def moving_average_filter(data, window_size=25):
    """
    Applies a moving average filter to the data.

    Parameters:
    data (array-like): The input signal.
    window_size (int): The size of the moving average window.

    Returns:
    array-like: The filtered signal.
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec



def data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]
            X_batch_reshaped = np.array(X_batch).reshape((-1, X_batch.shape[1], 1))
            yield (X_batch_reshaped, y_batch)


def load_data_chunk(X_train, y_train, weights_train, chunk_index, chunk_size):

    # Calculate the start and end indices of the chunk
    start = chunk_index * chunk_size
    end = start + chunk_size
    # Load a chunk of data and labels
    X_chunk = X_train[start:end]
    y_chunk = y_train[start:end]
    weights = weights_train[start:end]

    return X_chunk, y_chunk, weights