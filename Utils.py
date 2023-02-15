#importing necessary packages
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sklearn
from keras.layers import Conv1D, Dropout, Input, BatchNormalization, Flatten, Dense, Conv3DTranspose, UpSampling3D, Flatten, Reshape, Conv1D, LSTM, ConvLSTM1D
import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import pandas as pd
import ast
import itertools
import sys
import subprocess

# implement pip as a subprocess to install missing wfdb:
for pack in ['tensorflow', 'wfdb']:
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
  pack])
import wfdb
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler


def plot_ecg(ecg_dictionary, signals, peaks_df, beat_type = 'NORM', sampling_freq = 100, rec = 1, seg_len=100):

  '''
  Function to find and plot a random beat of a specified beat type in an ecg recording from the MIT database

  Args: 
  ecg_dictionary: Dict
    Dicitonary storing beat types

  procedure: string
    Name of the procedure from which we would like to plot. 

  beat_type: String 
    Specifies the beat type we would like to plot, from the annotations in the ecg_dictionary.

  Sampling Freq: Integer
    Deffaults to 100, sampling frequency of database, used to measure the time in the x axis. 

  rpeak: Int
    Deffaults to 1. Indicates the index of the rpeak (out of the rpeaks containing the type of beat specified)
    that we want to plot. 
  
  seg_len: Int
    Defaults to 100 and controls the size of the chunk of ECG we wish to plot, centered around the R-peak specified. 

  
  '''

  #finding random beat of specified type for plotting, and corresponding rpeak.
  #we find if the desired beat type is in the dictionary keys for the ecgs, which meansd the type of beat is present

  indexes_of_interest = [beat_type in ecg_dictionary['scp_codes'][i].keys() for i in ecg_dictionary['scp_codes'].keys()]
 
  #plot the desired recording index out of the recordings with specified arrhythmia
  ecg= signals[indexes_of_interest][rec]
  peaks = [int(peak) for peak in peaks_df['r_peaks'][indexes_of_interest].iloc[rec].split('[')[-1].split(']')[0].split()]
  
  #axis is the same length of the segment selected
  x_axis_complete=np.linspace(0, seg_len/sampling_freq , len(ecg))

  #plotting overall ecg
  plt.figure()
  plt.plot(x_axis_complete, ecg)

  #plotting rpeaks and surrounding area:
  half_segment = seg_len/2

  
  i=1
  for peak in peaks:
    sig_chunk = ecg[int(peak-half_segment) if peak>half_segment else 0: int(peak+half_segment) if int(peak+half_segment)<len(ecg) else -1 ]
    x_axis_chunk = np.linspace(0, seg_len/sampling_freq , len(sig_chunk))
 
    plt.figure()
    plt.title('Peak at sample ' + str(peak))
    plt.plot(x_axis_chunk, sig_chunk)
    plt.xlabel('Time (s)')

    i+=1
  plt.show()


def get_instant_heartrate(ecg_dictionary, signals, peaks_df, beat_type = 'NORM', sampling_freq = 100, rec = 1, seg_len=10):

  '''
  Function to find average instant heartrate for a certain beat type in the dataframe. 
  This function allows for windowing, so multiple instant heartrate measurements can be obtained from the same ecg strip.

  Args: 
  ecg_dictionary: Dict
    Dicitonary storing beat types, in our case from the database

  signals: array
    array containing all ecg strips.. 

  beat_type: String 
    Specifies the beat type we would like to plot, from the annotations in the ecg_dictionary. For example 'NORM' for a notmal beat.

  Sampling Freq: Integer
    Deffaults to 100, sampling frequency of database, used to calculate the time in the x axis. 

  rec: Int
    Deffaults to 1. Indicates the index of the record to show, out of the records identified with the desired beat type.
  
  seg_len: Int
    Defaults to 10 seconds, controls the length of the window used for instant heartrate.
  
  '''

  #finding all beats for the specified label

  indexes_of_interest = [beat_type in ecg_dictionary['scp_codes'][i].keys() for i in ecg_dictionary['scp_codes'].keys()]
  indexes_for_loop = [int(i) for i, x in enumerate(indexes_of_interest) if x]
  #fetch ecgs containing desired beat types
  ecg= signals[indexes_of_interest]

  average_hr = np.zeros([len(ecg), int(ecg.shape[-2]/ (seg_len * sampling_freq))])

  second_dim = int((ecg.shape[-2]/ (seg_len*sampling_freq)))
  #j indicates the ecg recording we are calculating averages for
  j=0
 
  for i in indexes_for_loop:
    #fetch peaks for the ecg
    
    peaks = np.array([int(peak) for peak in peaks_df['r_peaks'].iloc[i].split('[')[-1].split(']')[0].split()])

 

    #q indicates in which window or subsegment of that ecg we are at

    start=0
    
    for i in range(0, second_dim):
    
      window = peaks[start :] - peaks[start]
      #count, by the peak indexes, how much time has elapsed in this window, and take only the peaks that are withing the window time
      peaks_in_window = np.array(window[window.cumsum() <= seg_len *sampling_freq])
  
      #get mean r-r interval in seconds
      mean_diff= np.mean(np.diff(peaks_in_window))
      
      #storing beat, exception case for when we have more than one window present so array is two-dimensional
      if second_dim <= 1:
        average_hr[j] = mean_diff/sampling_freq
      else:
        average_hr[j, i] = mean_diff/sampling_freq
     

      #update starting peak for next window
      start += len(peaks_in_window.flatten())

    j+=1
    
  
  #divide length of window in seconds by time between beats to obtain beats per second
  beats_per_window = seg_len/average_hr
  
  #convert to beats per minute
  bpm= (60 * seg_len) / beats_per_window 
      
  return bpm  

def get_power_spect(ecg_dictionary, signals, peaks_df, beat_type = 'NORM', sampling_freq = 100):

  '''
  Function to find the power spectrums of ecgs of a specific beat type . 

  Args: 
  ecg_dictionary: Dict
    Dicitonary storing beat types

  procedure: string
    Name of the procedure from which we would like to calculate heartrate for. 

  beat_type: String 
    Specifies the beat type we would like to plot, from the annotations in the ecg_dictionary.

  Sampling Freq: Integer
    Deffaults to 100Hz, sampling frequency of database, used to measure the time in the x axis and important for frequnecy resolution. 
  
  '''

  #finding all beats for the specified label

  indexes_of_interest = [beat_type in ecg_dictionary['scp_codes'][i].keys() for i in ecg_dictionary['scp_codes'].keys()]
  ecg= signals[indexes_of_interest]


  #j indicates the ecg recording we are calculating the spectrum for
  j=0
 
  spect=np.zeros_like(ecg)

  for signal in ecg:

    for i in range(0, 11):
      #normalizing to compare power spectrums across channels of potentially different amplitudes
      signal[:,i] = signal[:,i]-np.min(signal[:,i])
      signal[:,i] = signal[:,i]/np.max(signal[:,i])

      #obtaining and storing power
      spect[j, :, i] = abs(fft(signal[:,i]))

    j+=1

  #storing corresponding frequencies
  xf= x=fftfreq(ecg.shape[-2], 1/sampling_freq)
  
  return spect, xf
