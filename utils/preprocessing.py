import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pywt import wavedec, waverec

from scipy.signal import butter, lfilter, savgol_filter


def butter_bandpass_filter(X_signal, filter_freq=0.55, fs=100, order=5):
    """
    This filter eliminates certain frequencies. We use it with a low frequency to eliminate the baseline drift.
    More information:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    """
    X_new = np.zeros_like(X_signal)
    b, a = butter(order, filter_freq, 'highpass', fs=fs)
    for i, x in enumerate(X_signal):
        # print(i)
        X_new[i] = lfilter(b, a, x, axis=0)
    return X_new


def smooth_signal(X_signal):
    """
    This function smoothes the signal with the savgol filter. 
    It approximates nearby points with a polynomial to find the smoothed line.
    """
    # Here more information: https://python.plainenglish.io/my-favorite-way-to-smooth-noisy-data-with-python-bd28abe4b7d0
    X_new = np.zeros_like(X_signal)
    for i in range(X_signal.shape[0]):
        for k in range(X_signal.shape[2]):
            X_new[i,:,k] = savgol_filter(X_signal[i,:,k], window_length=7, polyorder=2)
    return X_new


def normalize_signal(X_signal):
    """
    This function normalizes the signal to have a range of 1 and zero mean.
    """
    X_new = np.zeros_like(X_signal)
    for i in range(X_signal.shape[0]):
        for k in range(X_signal.shape[2]):
            x = X_signal[i,:,k]
            maximum = np.max(x)
            minimum = np.min(x)
            x = (x)/(maximum - minimum + 1e-8) 
            #X_signal[i,:,k] = x + (1 - np.max(x))
            X_new[i,:,k] = x - np.mean(x)
    return X_new


def denoise_signal(X_signal, dwt_transform='bior4.4', dlevels=9, cutoff_low=1, cutoff_high=8):
    X_new = np.zeros_like(X_signal)
    for i, x in enumerate(X_signal):
        # print(i)
        coeffs = wavedec(x, dwt_transform, level=dlevels)   # wavelet transform 'bior4.4'
        # scale 0 to cutoff_low
        for ca in range(0,cutoff_low):
            coeffs[ca]=np.multiply(coeffs[ca],[0.0])
        # scale cutoff_high to end
        for ca in range(cutoff_high, len(coeffs)):
            coeffs[ca]=np.multiply(coeffs[ca],[0.0])
        X_new[i] = waverec(coeffs, dwt_transform) # inverse wavelet transform
    return X_new


def median_filter(X_signal, window_size=3):
    X_new = np.zeros_like(X_signal)
    for i, x_i in enumerate(X_signal):
        for j, x in enumerate(x_i):
            filtered_data = np.zeros(len(x))
            for k in range(len(x)):
                start = max(0, k - window_size // 2)
                end = min(len(x), k + window_size // 2 + 1)
                window = x[start:end]
                filtered_data[k] = np.median(window)
            X_signal[i][j] = filtered_data
    return X_signal



