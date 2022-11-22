import sys
import os
import enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt
import wfdb


_DB_NAME =  "ptbxl_database.csv"


class Leads(enum.Enum):
    I = "I"
    II = "II"
    III = "III"
    AVR = "AVR"
    AVL = "AVL"
    AVF = "AVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"


class Freq(enum.Enum):
    LOW = enum.auto()
    HIGH = enum.auto()


SEX_TO_STR = {0: "Male", 1: "Female"}


def extract_signals(data_path: str,
                    ecg_id: int,
                    leads: Optional[List[Leads]],
                    freq: Freq) -> Tuple[pd.Series, np.ndarray]:
    ecg_entry = pd.read_csv(os.path.join(data_path, _DB_NAME)) \
                  .set_index("ecg_id") \
                  .loc[ecg_id]
    data_file = ecg_entry[f"filename_{'hr' if freq == Freq.HIGH else 'lr'}"]
    record: wfdb.Record = wfdb.rdrecord(
        record_name=os.path.join(data_path, data_file),
        channel_names=[
            lead._value_ for lead in (leads if leads is not None else Leads)
        ],
    )
    ecg_entry["frequency"] = record.fs
    return ecg_entry, record.p_signal


def plot_ecg(data_path: str,
             ecg_id: int,
             lead: Leads,
             freq: Freq):
    ecg_entry, signal = extract_signals(data_path, ecg_id, [lead], freq)
    record_freq = ecg_entry.frequency

    _, ax = plt.subplots()
    r_peaks = parse_r_peaks_entry(ecg_entry["r_peaks"])
    for peak in r_peaks:
        ax.axvline(peak / record_freq, linestyle="--", color="lightgray")

    ax.plot(np.arange(len(signal)) / record_freq,
            signal,
            linestyle='dashed',
            linewidth="1",
            color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (mV)")

    plt.title(
        f"Signal: {lead._value_} | "
        f"ECG: {ecg_id} | "
        f"Patient: {int(ecg_entry.patient_id)} | "
        f"Age: {int(ecg_entry.age)} | "
        f"Sex: {SEX_TO_STR[int(ecg_entry.sex)]}"
    )
    plt.show()


def parse_r_peaks_entry(entry: str):
    return [int(x) for x in entry[1:-1].split(" ") if x != ""]


def norm_and_denoise(signals: np.ndarray,
                     sing_values: int):
    new_signals = detrend(signals, axis=0)
    new_signals = new_signals - np.mean(new_signals, axis=0)[np.newaxis, :]
    u, s, v = np.linalg.svd(new_signals)
    s[sing_values - len(s):] = 0.
    cleaned = np.matmul(u[..., :len(s)] * s[..., None, :], v)
    return cleaned


def find_r_frequency(signals: np.ndarray, frequency: float) -> Tuple[List[int], float]:
    signals = np.square(signals)

    all_peaks = []
    for i in range(signals.shape[1]):
        signal = signals[:, i]
        signal = signal / (np.max(signal) - np.min(signal))
        peaks_idx, _ = find_peaks(signal, prominence=0.1)
        all_peaks = all_peaks + list(peaks_idx)

    counts = {i: all_peaks.count(i) for i in all_peaks}
    peaks_idx = sorted([i for i in counts.keys() if counts[i] > 5])
    return peaks_idx, len(peaks_idx) / (len(signals) / frequency)
