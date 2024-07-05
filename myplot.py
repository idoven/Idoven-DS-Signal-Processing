# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurokit2.misc import NeuroKitWarning
from neurokit2.signal.signal_rate import _signal_rate_plot
from neurokit2.ecg.ecg_peaks import _ecg_peaks_plot
from neurokit2.ecg.ecg_segment import ecg_segment


from neurokit2.events import events_plot
from neurokit2.stats import standardize as nk_standardize


def ecg_plot(ecg_signals, info=None):
    """**Visualize ECG data**

    Plot ECG signals and R-peaks.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from ``ecg_process()``.
    info : dict
        The information Dict returned by ``ecg_process()``. Defaults to ``None``.

    See Also
    --------
    ecg_process

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: python

      # To be run after ecg_plot()
      fig = plt.gcf()
      fig.set_size_inches(10, 12, forward=True)
      fig.savefig("myfig.png")

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)

      # Process signal
      signals, info = nk.ecg_process(ecg, sampling_rate=1000)

      # Plot
      @savefig p_ecg_plot.png scale=100%
      nk.ecg_plot(signals, info)
      @suppress
      plt.close()

    """
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    if info is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category=NeuroKitWarning,
        )
        info = {"sampling_rate": 1000}

    # Extract R-peaks (take those from df as it might have been cropped)
    if "ECG_R_Peaks" in ecg_signals.columns:
        info["ECG_R_Peaks"] = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    gs = matplotlib.gridspec.GridSpec(10, 6, width_ratios=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6,1/6])

    fig = plt.figure(figsize=(12, 8),constrained_layout=False)
    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")

    ax0 = fig.add_subplot(gs[0:4, 0:3])
    ax1 = fig.add_subplot(gs[6:10, 0:3], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, 4:])

    # Plot signals
    phase = None
    if "ECG_Phase_Ventricular" in ecg_signals.columns:
        phase = ecg_signals["ECG_Phase_Ventricular"].values
	
    ax0 = _ecg_peaks_plot(
        ecg_signals["ECG_Clean"].values,
        info=info,
        sampling_rate=info["sampling_rate"],
        raw=ecg_signals["ECG_Raw"].values,
        quality=ecg_signals["ECG_Quality"].values,
        phase=phase,
        ax=ax0,
    )
    ax0.legend(loc='upper center', bbox_to_anchor=(1.15, 1), ncol=1)

    # Plot Heart Rate
    ax1 = _signal_rate_plot(
        ecg_signals["ECG_Rate"].values,
        info["ECG_R_Peaks"],
        sampling_rate=info["sampling_rate"],
        title="Heart Rate",
        ytitle="Beats per minute (bpm)",
        color="#FF5722",
        color_mean="#FF9800",
        color_points="#FFC107",
        ax=ax1,
    )

    # Plot individual heart beats
    ax2 = ecg_segment(
        ecg_signals,
        info["ECG_R_Peaks"],
        info["sampling_rate"],
        show="return",
        ax=ax2,
    )



def signal_plot(
    signal, sampling_rate=None, subplots=False, standardize=False, labels=None, **kwargs
):
    """**Plot signal with events as vertical lines**

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Needs to be supplied if
        the data should be plotted over time in seconds. Otherwise the data is plotted over samples.
        Defaults to ``None``.
    subplots : bool
        If ``True``, each signal is plotted in a subplot.
    standardize : bool
        If ``True``, all signals will have the same scale (useful for visualisation).
    labels : str or list
        Defaults to ``None``.
    **kwargs : optional
        Arguments passed to matplotlib plotting.

    See Also
    --------
    ecg_plot, rsp_plot, ppg_plot, emg_plot, eog_plot

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: console

        # To be run after signal_plot()
        fig = plt.gcf()
        fig.savefig("myfig.png")

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, sampling_rate=1000)
      @savefig p_signal_plot1.png scale=100%
      nk.signal_plot(signal, sampling_rate=1000, color="red")
      @suppress
      plt.close()

    .. ipython:: python

       # Simulate data
      data = pd.DataFrame({"Signal2": np.cos(np.linspace(start=0, stop=20, num=1000)),
                           "Signal3": np.sin(np.linspace(start=0, stop=20, num=1000)),
                           "Signal4": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000)))})

      # Process signal
      @savefig p_signal_plot2.png scale=100%
      nk.signal_plot(data, labels=['signal_1', 'signal_2', 'signal_3'], subplots=True)
      nk.signal_plot([signal, data], standardize=True)
      @suppress
      plt.close()

    """
    # Sanitize format
    if isinstance(signal, list):
        try:
            for i in signal:
                len(i)
        except TypeError:
            signal = np.array(signal)

    if isinstance(signal, pd.DataFrame) is False:

        # If list is passed
        if isinstance(signal, list) or len(np.array(signal).shape) > 1:
            out = pd.DataFrame()
            for i, content in enumerate(signal):
                if isinstance(content, pd.Series):
                    out = pd.concat(
                        [out, pd.DataFrame({content.name: content.values})],
                        axis=1,
                        sort=True,
                    )
                elif isinstance(content, pd.DataFrame):
                    out = pd.concat([out, content], axis=1, sort=True)
                else:
                    out = pd.concat(
                        [out, pd.DataFrame({"Signal" + str(i + 1): content})],
                        axis=1,
                        sort=True,
                    )
            signal = out

        # If vector is passed
        else:
            signal = pd.DataFrame({"Signal": signal})

    # Copy signal
    signal = signal.copy()

    # Guess continuous and events columns
    continuous_columns = list(signal.columns.values)
    events_columns = []
    for col in signal.columns:
        vector = signal[col]
        if vector.nunique() == 2:
            indices = np.where(vector == np.max(vector.unique()))
            if bool(np.any(np.diff(indices) == 1)) is False:
                events_columns.append(col)
                continuous_columns.remove(col)

    # Adjust for sampling rate
    if sampling_rate is not None:
        signal.index = signal.index / sampling_rate
        title_x = "Time (seconds)"
    else:
        title_x = "Time"
    #        x_axis = np.linspace(0, signal.shape[0] / sampling_rate, signal.shape[0])
    #        x_axis = pd.DataFrame(x_axis, columns=["Time (s)"])
    #        signal = pd.concat([signal, x_axis], axis=1)
    #        signal = signal.set_index("Time (s)")

    # Plot accordingly
    if len(events_columns) > 0:
        events = []
        for col in events_columns:
            vector = signal[col]
            events.append(np.where(vector == np.max(vector.unique()))[0])
        events_plot(events, signal=signal[continuous_columns])
        if sampling_rate is None and pd.api.types.is_integer_dtype(signal.index):
            plt.gca().set_xlabel("Samples")
        else:
            plt.gca().set_xlabel(title_x)

    else:

        # Aesthetics
        colors = [
            "b",
            "b",
            "b",
            "b",
            "b",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
        ]
        
        #if len(continuous_columns) > len(colors):
         #   colors = plt.cm.viridis(np.linspace(0, 1, len(continuous_columns)))

        # Plot
        if standardize is True:
            signal[continuous_columns] = nk_standardize(signal[continuous_columns])

        if subplots is True:
            fig, axes = plt.subplots(nrows=len(continuous_columns), ncols=1, sharex=True, figsize=(20, 15))
            fig.subplots_adjust(hspace=0.5, wspace=0.4)
            for ax, col in zip(axes, continuous_columns):
                ax.plot(signal[col], c='b', **kwargs)
                ax.grid(True)
        else:
            _ = signal[continuous_columns].plot(subplots=False, sharex=True, **kwargs)

        if sampling_rate is None and pd.api.types.is_integer_dtype(signal.index):
            plt.xlabel("Samples")
        else:
            plt.xlabel(title_x)

    # Tidy legend locations and add labels
    if labels is None:
        labels = continuous_columns.copy()

    if isinstance(labels, str):
        n_labels = len([labels])
        labels = [labels]
    elif isinstance(labels, list):
        n_labels = len(labels)

    if len(signal[continuous_columns].columns) != n_labels:
        raise ValueError(
            "NeuroKit error: signal_plot(): number of labels does not equal the number of plotted signals."
        )

    if subplots is False:
        plt.legend(labels, loc=1)
    else:
        for i, label in enumerate(labels):
            axes[i].legend([label], loc=1)
