import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, lfilter
import wfdb



def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a

def highpass(data, fs, cutoff_high, order=5):
    b,a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b,a,data)
    return x

def plot_meanstd_byColumn(df, sampling_f):

    array_mean = []
    array_std  = []

    for i in range(df.shape[1]):  # rows = df.shape[0], cols = df.shape[1]
    
        mean_val = df[i].describe()[1] # mean value
        std_val  = df[i].describe()[1] # std value

        array_mean.append(mean_val)
        array_std.append(std_val)

    max_time   = np.shape(array_mean)[0]/sampling_f
    time_steps = np.linspace(0, max_time, np.shape(array_mean)[0]) * 1000
    
    plt.figure()
    plt.errorbar(x = time_steps, y = array_mean, yerr = array_std, ecolor='r')
    plt.title('Mean value ecg (+-Standard Deviation)')
    plt.tick_params(axis='x', which='major')
    plt.xlabel('Time[ms]')

    f = np.abs(np.fft.fft(array_mean))
    freq_steps = np.fft.fftfreq(np.shape(array_mean)[0], d=1/sampling_f)
    plt.figure()
    plt.plot(freq_steps, f)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")

    