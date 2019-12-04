
import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt

plt.close("all")

# Spectrogram Generation Function
def generateSpectrogram(x, window_size, frame_rate, frame_skip):
    """ Generates spectrogram using scipy signal stft function, and returns magnitude spectrum
    
    Args:
        x (np.ndarray): Time series input signal, 1D array of integers
        window_size (int): Int with the length of the window (Hamming window is used)
        frame_rate (int): Frames per second
        frame_skip (int): Number of frames overlapping each frame
    
    Returns:
        freq_axis (np.ndarray): 1D array of freqency labels per freq bin
        time_axis (np.ndarray): 1D array of time labels per time bin
        mag_spectrum (np.ndarray): 2D array of signal magnitudes
        
    """
    
    # Generate Hamming window of length window_size
    window_func = np.hamming(window_size)

    # Generate spectrogram
    freq_axis, time_axis, Zxx = signal.stft(x, frame_rate, window=window_func, nperseg=window_size, noverlap = frame_skip)

    # Obtain magnitude spectrum
    mag_spectrum = np.abs(Zxx)
    
    return freq_axis, time_axis, np.sqrt(mag_spectrum)