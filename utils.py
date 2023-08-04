"""Utils to load and visualize data."""
import wfdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def load_raw_data(df: pd.DataFrame, data_path: str, sampling_rate: int = 500):
    """Load raw data based on sampling rate. 
    Args:
        df: dataframe containing filenames.
        sampling_rate: int, 100 or 500.
        data_path: path to the data."""
    if sampling_rate == 100:
        data = [
            wfdb.rdsamp(os.path.join(data_path, f)) for f in df.filename_lr
        ]
    else:
        data = [
            wfdb.rdsamp(os.path.join(data_path, f)) for f in df.filename_hr
        ]
    data = np.array([signal for signal, meta in data])
    return data


def one_hot_encode_diagnosis(scp_df: pd.DataFrame, metadata: pd.DataFrame):
    """One-hot encode diagnosis.
    Args:
        scp_df: dataframe containing diagnosis information.
        metadata: dataframe containing ecg information.
        """
    codes = metadata.scp_codes
    code_summary = {code: [] for code in np.unique(scp_df.index.values)}

    for ecg_code in codes:
        for target_code in code_summary.keys():
            if target_code in ecg_code.keys():
                code_summary[target_code].append(ecg_code[target_code])
            else:
                code_summary[target_code].append(np.nan)

    code_summary = pd.DataFrame(code_summary)
    code_summary.head()
    metadata = metadata.join(code_summary)
    return code_summary, metadata


def plot_fft(ecg: np.array, sampling_rate: int = 500):
    """Plot the FFT of the ecg signal.
    Args:
        ecg: numpy array containing the ecg signal.
        sampling_rate: sampling rate of the ecg signal, in Hz."""
    fft_result = np.fft.fft(ecg)
    t = np.linspace(0, 10, int(sampling_rate * 10))

    fft_magnitude = np.abs(fft_result)
    fft_frequency = np.fft.fftfreq(len(fft_result), d=1 / sampling_rate)
    plt.figure(figsize=(10, 5))
    plt.plot(fft_frequency, fft_magnitude)
    plt.title("FFT of ECG Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.yscale('log')
    plt.xlim(0, sampling_rate / 2)
    plt.tight_layout()
    plt.show()


def plot_pqrst(ecg: np.array, duration: int = 2, sampling_rate: int = 500):
    """Plot the PQRST complex of the ecg signal.
    Args:
        ecg: numpy array containing the ecg signal.
        duration: duration of the plot in seconds.
        sampling_rate: sampling rate of the ecg signal, in Hz."""
    plt.figure(figsize=(10, 5))
    time = np.arange(0, duration, 1 / sampling_rate)
    plt.plot(time, ecg[0:sampling_rate * duration])
    plt.xlabel("time(s)")
    plt.title("PQRST complex")
    plt.show()


def plot_pqrst_comparison(ecg_normal: np.array,
                          ecg_abnormal: np.array,
                          duration: int = 2,
                          sampling_rate: int = 500,
                          labels: List[str] = ["normal", "abnormal"]):
    """Compare the PQRST complex of two ecg signals.
    Args:
        ecg_normal: numpy array containing the normal ecg signal.
        ecg_abnormal: numpy array containing the abnormal ecg signal.
        duration: duration of the plot in seconds.
        sampling_rate: sampling rate of the ecg signal, in Hz.
        labels: list of labels for the two ecg signals."""

    plt.figure(figsize=(10, 5))
    time = np.arange(0, duration, 1 / sampling_rate)
    plt.plot(time, ecg_normal[0:sampling_rate * duration], label=labels[0])
    plt.plot(time, ecg_abnormal[0:sampling_rate * duration], label=labels[1])
    plt.xlabel("time(s)")
    plt.title("PQRST complex")
    plt.legend()
    plt.show()
