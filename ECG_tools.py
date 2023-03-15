#!pip install heartpy
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import ecg_plot
import heartpy as hp

#-----------------------------------------------------------------------------------------------------------------------------------
# from the example_physionet.py
def load_raw_data(metadata, sampling_rate, path):
    # if downsampled records are wanted 
    if sampling_rate == 100:
        # take 'filename_lr' from the metadata of the database (downsampled versions of the records are stored in filename_lr)
        data = [wfdb.rdsamp(path + f) for f in metadata['filename_lr']]
    # if original records are wanted
    else: 
        # take 'filename_hr' from the metadata of the database (the original records are stored in filename_hr)
        data = [wfdb.rdsamp(path + f) for f in metadata['filename_hr']]
    # take only ECG data from the dictionary
    data = np.array([signal for signal, meta in data])
    
    return data
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# Define a function to get diagnostic classes and descriptions where it is pointed in the scp_statements.csv file (e.g. NORM refers to normal ECG)
def aggregate_diagnostic(y_dic, scp_data):

    y_dic = np.asarray(y_dic) # convert input to list
    diagnostic_superclass = [None] * len(y_dic) # define an empty diagnostic_superclass list to assign relevant scp_codes descriptions
    diagnostic_class = [None] * len(y_dic) # define an empty diagnostic_class list to assign relevant scp_codes diagnostic classes
    scp_codes_incides = list(scp_data.index) # convert indices to list

    for j in range (0,len(y_dic)):

        key = list(y_dic[j]) # get key values

        for i in range(0, len(key)):

            if key[i] in scp_codes_incides: # if teh scp code is define in the scp_statements.csv file
   
                diagnostic_class[j] = scp_data['diagnostic_class'][key[i]] # get the relevant diagnostic class
                diagnostic_superclass[j] = scp_data['description'][key[i]] # get the relevant description

    return diagnostic_class, diagnostic_superclass
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
def plot_ecg(ecg, how_many, metadata, diag_class, diagnostics):

    for i in range(0,how_many):

        if ecg[i].shape == (1000,12):
            
            ecg_plot.plot(np.transpose(ecg[i]), sample_rate = 100, title = ['Patient ID:', metadata['patient_id'].iloc[i], 'Diagnostic Class: ', diag_class[i], 'Diagnostic: ', diagnostics[i], 'Age: ', metadata['age'].iloc[i]])

            
        else: 
            ecg_plot.plot(ecg[i], sample_rate = 100, title = ['Patient ID:', metadata['patient_id'].iloc[i], 'Diagnostic Class: ', diag_class[i], 'Diagnostic: ', diagnostics[i], 'Age: ', metadata['age'].iloc[i]])
        
            return ecg_plot.show() 
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# Define a function to get R peaks for desired number of EGCs
def get_r_peaks(data, how_many, metadata):

    r_peaks = []
    calculated_r_peak_values = []
    ecg = []

    for i in range(0, how_many):

        r_peaks_from_database = metadata['r_peaks'].iloc[i] # get given R peak indices from metadata
        r_peaks_from_database = str.split(r_peaks_from_database[1:-1].strip()) # split values to iterate
        r_peaks_from_database = [int(numeric_string) for numeric_string in r_peaks_from_database] # convert given R peak indices as string to integer
        r_peaks.append(r_peaks_from_database)

        temp_ecg = data[i][:,0] # get ECG signal
        ecg.append(temp_ecg)
        r_peak_values = temp_ecg[r_peaks_from_database] # get R peak values 
        
        calculated_r_peak_values.append(r_peak_values)

    return r_peaks, calculated_r_peak_values, ecg
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# Define a function to plot R peaks on relevant ECG with patient ID
def plot_r_peaks(data, rPeaks, rPeakValues, metadata):

    for r in range(0,len(data)):
       
        plt.figure()
        plt.title('R Peaks for the Patient ID with ' + str(metadata['patient_id'].iloc[r]))
        plt.plot(data[r])
        plt.plot(rPeaks[r], rPeakValues[r], marker='o', ls='', ms=3)
        plt.xlabel('Time (s)')
        plt.ylabel('mV')
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# Define a function to calculate heart beat, total heart beat and average heart beat for desired number of ECGs
def calculate_HRV(data, how_many, metadata):

    heart_beat = []
    total_heart_beat = []
    average_heart_beat = []
    indices = []
    heart_rate_variablity = pd.DataFrame()

    for i in range(0, how_many):
        
        fs = 100 # sampling rate
        temp_ecg = data[i][:,0] # get ECG

        working_data, measures = hp.process(temp_ecg, fs)
        hb = measures['bpm'] # get heart beat from measures
        tothb = len(working_data['peaklist']) # total heart beat equals to number of R peaks
        avhb = hb / tothb # average heart beat equals to heart beat/total heart beat

        heart_beat.append(hb)
        total_heart_beat.append(tothb)
        average_heart_beat.append(avhb) 
        indices.append(metadata['patient_id'].iloc[i])

    # Define a data frame to see all calculated values
    heart_rate_variablity['Patient ID'] = indices
    heart_rate_variablity['Heart Beat (bpm)'] = heart_beat
    heart_rate_variablity['Total Heart Beat'] = total_heart_beat
    heart_rate_variablity['Average Heart Beat'] = average_heart_beat
            
    return heart_rate_variablity
#-----------------------------------------------------------------------------------------------------------------------------------
