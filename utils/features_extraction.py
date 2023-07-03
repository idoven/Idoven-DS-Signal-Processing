import pandas as pd
import numpy as np
import ast
import wfdb

from scipy import signal
import wfdb
import wfdb.processing
import ecg_plot


def compute_rpeaks(waveform, Fs, threshold_ratio=0.7):
    timeEGC = np.arange(waveform.shape[0])*1/Fs
    interval = max(waveform) - min(waveform)
    threshold = threshold_ratio*interval + min(waveform)
    
    maxima = []
    maxima_indices = []
    mxs_indices = []
    banner = False
    
    for i in range(0, len(waveform)):
            
        if waveform[i] >= threshold:#If a threshold value is surpassed,
            # the indices and values are saved 
            banner = True
            maxima_indices.append(i)
            maxima.append(waveform[i])
            
        elif banner == True and waveform[i] < threshold: #If the threshold value is crossed
            # the index of the maximum value in the original array is saved
            index_local_max = maxima.index(max(maxima))
            mxs_indices.append(maxima_indices[index_local_max])
            maxima = []
            maxima_indices = []
            banner = False     

    return mxs_indices

def compute_dist_rpeaks(mxs_indices, waveData):
    mins_try = []
    i=0

    for R_peak_i in mxs_indices:
        left_interval = waveData[R_peak_i-10:R_peak_i]
        right_interval = waveData[R_peak_i:R_peak_i+10]
        mins_try.append(R_peak_i - 10 + (list(left_interval).index(min(left_interval))) )
        mins_try.append(R_peak_i + (list(right_interval).index(min(right_interval))))
    return mins_try

def compute_heart_rate(waveform, Fs=100):
    time_wf = np.arange(waveform.shape[0])*1/Fs
    rpeaks = compute_rpeaks(waveform, Fs)
    heart_rate = Fs * (60.0 / np.diff(rpeaks))
    
    return heart_rate

def bandpass_waveform(waveform,Fs):
    W1     = 5*2/Fs                                    
    W2     = 15*2/Fs                                  
    b, a   = signal.butter(4, [W1,W2], 'bandpass')     
    waveform    = np.asarray(waveform)                          
    waveform    = np.squeeze(waveform)                           
    waveform_BP = signal.filtfilt(b,a,waveform)    
    return waveform_BP

def differentiate_waveform(waveform):
    '''
    Compute single difference of the signal ECG
    '''
    waveform_df  = np.diff(waveform)
    waveform_sq  = np.power(waveform_df,2)
    return np.insert(waveform_sq,0, waveform_sq[0])

def movingaverage_waveform(waveform):
    
    N = int(0.03 * int(100)) 
    window  = np.ones((1,N))/N
    waveform_ma  = np.convolve(np.squeeze(waveform),np.squeeze(window))
    return waveform_ma

def QRS_peaks(waveform,Fs):
    
    wf_bp = bandpass_waveform(waveform,Fs)
    wf_diff = differentiate_waveform(wf_bp)
    wf_ma = movingaverage_waveform(wf_diff)
    
    peaks, _  = signal.find_peaks(wf_ma, height=np.mean(wf_ma), distance=round(Fs*0.200))
    return wf_bp, wf_diff, wf_ma, peaks