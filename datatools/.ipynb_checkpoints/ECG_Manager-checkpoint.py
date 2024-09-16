import utils as ut
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import math

class ECG_Manager:
    """ Main class to manage the data of the ECG dataset"""
    
    def __init__(self, annotations, path = "", sampling_rate = 100):
        """ Initialization method
        
        Args:
        - annotations: annotations of the ECG dataset
        - path: path of the downloaded dataset
        - sampling_rate
        """
        
        self.notes = annotations.copy()
        self.path = path
        self.fs = sampling_rate
        self.data = ut.load_raw_data(annotations, sampling_rate, path)
        self.channel_names = ["I", "II", "III", "AVL", "AVR", "AVF", "V1","V2", "V3", "V4", "V5", "V6"]
        
        self.notes["total_heart_beats"] = np.nan
        self.notes["heart_rate"] = np.nan
        self.notes.index = self.notes.index-1
        
    # VISUALIZATION METHODS
    def plot_epoch(self, ecg: int, channels = [1], baseline_correction = True, force_baseline_correction = False):
        """ Function to visualize epochs from the data
        Args:
        - ecg (int): epoch from the dataset that will be plot
        - channels (list or int): channels to plot
        - baseline_correction: correct the baseline with a highpass filter in case a baseline_drift was marked
        - force_baseline_correction: correct the baseline independently of the baseline_drif mark
        """
        
        # Transform to list in case only one channel was passed as int
        if type(channels) == int:
            channels = [channels]
        if channels == "all":
            channels = list(np.arange(0, len(self.channel_names)).astype(int))
            
        epoch = self.data[ecg, :, :]
        
        # Remove baseline if indicated by the args
        if baseline_correction and (isinstance(self.notes.baseline_drift[ecg], str) or force_baseline_correction):
            epoch = self.remove_baseline(epoch)
        
        # Vector time for the x axis of the plot
        vTime = np.linspace(0, epoch.shape[0]/self.fs, epoch.shape[0])
        
        # Plotting
        if len(channels) == 1:
            plt.plot(vTime, epoch[:, channels])
            plt.ylabel(self.channel_names[channels[0]] + " (mV)")
            plt.xlabel("Time (s)")
        else:
            if len(channels) > 5:
                nrows = np.ceil(len(channels)/2).astype(int)
                ncolumns = 2
                fig, axs = plt.subplots(nrows, 2)
            else:
                nrows = len(channels)
                ncolumns = 1
                fig, axs = plt.subplots(len(channels))
            plt_index = 0
            for c in channels:
                axs[np.unravel_index(plt_index, (nrows,ncolumns))].plot(vTime, epoch[:, c])
                axs[np.unravel_index(plt_index, (nrows,ncolumns))].set(ylabel = self.channel_names[c] + " (mV)")
                plt_index = plt_index + 1
            
            fig.suptitle("Epoch " + str(ecg))
            fig.supxlabel("Time (s)")
            fig.tight_layout()
            fig.set_figheight(10)
            fig.set_figwidth(15)
            
    # PREPROCESS METHODS (if needed)
    def remove_baseline(self, epoch, freq = 0.5):
        """ Method to correct the baseline drift from the epochs
        Args:
        - epoch: data to be corrected
        - freq: cutoff frequency for the filter
        """
        
        b, a = sig.butter(3, freq, 'highpass', fs = self.fs)
        filtered = sig.filtfilt(b, a, epoch, axis = 0)
        return filtered
    
    # FEATURE EXTRACTION METHODS
    def compute_peaks(self, ecg: int, thresh_prop = 0.6):
        """ Method to compute the peaks from the epoch. The algorithm uses the 
        channel with the maximum value (better distinction between R and T peaks)
        
        Args:
        - ecg: index of epoch
        - thresh_prop: proportion of the maximum value to be used as threshold
        
        NOTE: I know that the original annotation file contains r_peaks, I am just
        implementing my own method because I was unsure if it was part of the test.
        """
        
        epoch = self.data[ecg, :, :]
        epoch = self.remove_baseline(epoch)
        
        maxPeak_channel = np.unravel_index(np.argmax(epoch), epoch.shape)[1]
        epoch = epoch[:, maxPeak_channel]
        
        peaks_pos, props = sig.find_peaks(epoch, height = np.max(epoch)*thresh_prop)
        peaks_pos = peaks_pos
        return peaks_pos                                   
        
    def compute_hr(self, ecg: int):
        """ Calculation of the heart rate and annotiation into the notes
        DataFrame. The heart rate is computed as the inverse of the median
        RR interval (increased robustness against momentaneous pathologies
        and false positives or negatives from the peak detector"""
        
        peaks_pos = self.compute_peaks(ecg)/self.fs
        median_rr_interval = np.median(np.diff(peaks_pos))
        
        heart_rate = 60/median_rr_interval
        self.notes.heart_rate[ecg] = heart_rate
        return heart_rate
    
    def compute_thb(self, ecg:int):
        """ Calculation of total heart beats by counting the number
        of detected peaks"""
        
        peaks_pos = self.compute_peaks(ecg)/self.fs
        
        total_heart_beats = peaks_pos.shape[0]
        self.notes.total_heart_beats[ecg] = total_heart_beats
        return total_heart_beats
        
        
                                    
                                    