import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ecg_signal(X_patient, channels: list, sampling_rate: int, subfig_height: float = 3,
                    subfig_width: float = 20, heartbeats: np.array = None, 
                    heartbeat_peaks: np.array = None, qrs_intervals: np.array = None):
    time_in_s = np.linspace(0, X_patient.shape[0] / sampling_rate, X_patient.shape[0])

    rows = len(channels)
    fig, axes = plt.subplots(rows, sharey='row')

    fig_height = subfig_height*rows

    for i, channel in enumerate(channels):
        if i == 0:
            ecg_label = 'ECG Signal'
            heartbeat_peaks_label = 'Detected heartbeat peaks.'
        else:
            ecg_label = None
            heartbeat_peaks_label = None
        axes[i].plot(time_in_s, X_patient[:, i], label=ecg_label)
        axes[i].set(ylabel=channel)
        axes[i].grid()
        if heartbeat_peaks is not None:
            axes[i].plot(time_in_s[heartbeat_peaks[i]], X_patient[heartbeat_peaks[i], i], "r^", markersize=10, 
                         label=heartbeat_peaks_label)
        if heartbeats is not None:
            for j, h in enumerate(heartbeats):
                if i == 0 and j == 0:
                    heartbeat_label = 'Estimated Heartbeats.'
                else:
                    heartbeat_label = None
                axes[i].axvline(time_in_s[h], color = 'r', label = heartbeat_label, alpha=0.5)
        if qrs_intervals is not None:
            qrs_starts, qrs_ends = qrs_intervals 
            for j in range(len(qrs_starts)):
                if i == 0 and j == 0:
                    qrs_label = 'Estimated QRS complex.'
                else:
                    qrs_label = None
                axes[i].axvspan(time_in_s[int(qrs_starts[j])], time_in_s[int(qrs_ends[j])], 
                                alpha=0.5, color='yellow', label=qrs_label)
                
    fig.legend(loc='upper center')
    plt.setp(axes, xticks=np.arange(min(time_in_s), max(time_in_s), 0.2), yticks=np.arange(-1.5, 2.0, 0.5))
    plt.ylim(-1.5, 2.0)

    fig.supxlabel("Time in seconds")
    fig.supylabel("mV")

    fig.set_figheight(fig_height)
    fig.set_figwidth(subfig_width)
    plt.grid(True)
    
    plt.show()