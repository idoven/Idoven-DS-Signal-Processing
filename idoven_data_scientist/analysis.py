import sys
import os
import enum
import numpy as np
import pandas as pd
import wfdb

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtagg')


_DB_NAME =  "ptbxl_database.csv"


class Freq(enum.Enum):
    LOW = enum.auto()
    HIGH = enum.auto()

    def from_str(freq_str: str):
        pass


def plot_ecg(data_path: str, ecg_id: int, freq: Freq = Freq.LOW):
    db: pd.DataFrame = \
        pd.read_csv(os.path.join(data_path, _DB_NAME)) \
          .set_index("ecg_id")
    ecg_entry = db.loc[int(ecg_id)]
    data_file = ecg_entry[f"filename_{'hr' if freq == Freq.HIGH else 'lr'}"]
    record: wfdb.Record = wfdb.rdrecord(
        record_name=os.path.join(data_path, data_file),
        channel_names=["I"],
    )
    signal = record.p_signal

    _, ax = plt.subplots()
    ax.plot(np.arange(len(signal)) / record.fs, signal, marker='.', linestyle='dashed', color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (mV)")

    r_peaks = [int(x) for x in ecg_entry["r_peaks"][1:-1].split(" ") if x != ""]
    for peak in r_peaks:
        ax.axvline(peak / record.fs, linestyle="--", color="k")

    plt.show()


def analyse_single_record(path: str):
    record: wfdb.Record = wfdb.rdrecord(path)
    wfdb.plot_wfdb(record)


if __name__ == "__main__":
    plot_ecg(*sys.argv[1:])
