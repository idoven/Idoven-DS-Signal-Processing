import sys
import os
import enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
import matplotlib.pyplot as plt
import wfdb
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score


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
    db = pd.read_csv(os.path.join(data_path, _DB_NAME)).set_index("ecg_id")
    return _extract_from_db(db, data_path, ecg_id, leads, freq)


def _extract_from_db(db: pd.DataFrame,
                     data_path: str,
                     ecg_id: int,
                     leads: Optional[List[Leads]],
                     freq: Freq) -> Tuple[pd.Series, np.ndarray]:
    ecg_entry = db.loc[ecg_id].copy()
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


def extract_train_data(num_samples: int, train_test_prop: float) \
        -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    samples = []
    labels = []
    data_path = "./data/physionet.org/files/ptb-xl/1.0.2/"
    db = pd.read_csv(os.path.join(data_path, _DB_NAME)).set_index("ecg_id")

    for i in range(1, num_samples + 1):
        if i not in db.index:
            continue
        ecg_entry, raw_signals = _extract_from_db(
            db=db,
            data_path=data_path,
            leads=None,
            ecg_id=i,
            freq=Freq.LOW,
        )
        cleaned = norm_and_denoise(raw_signals, sing_values=4)
        samples.append(cleaned)
        peaks_lbl = np.zeros(cleaned.shape[0], dtype=int)
        peaks_lbl[parse_r_peaks_entry(ecg_entry.r_peaks)] = 1
        labels.append(peaks_lbl)

    samples = np.array(samples)
    samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0] * labels.shape[1])

    dataset_len = len(labels)
    train_start = int(dataset_len * train_test_prop)

    # (train_samples, train_labels), (test_samples, test_labels)
    return (samples[:train_start, :], labels[:train_start]), (samples[train_start:, :], labels[train_start:])


def train_and_test(dataset: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]) \
        -> RidgeClassifier:
    (tr_samples, tr_labels), (te_samples, te_labels) = dataset
    classifier = RidgeClassifier()
    classifier.fit(tr_samples, tr_labels)
    score = classifier.score(te_samples, te_labels)
    print(
        f"Model finalised training and testing with a mean accuracy of {score}."
    )
    return classifier
