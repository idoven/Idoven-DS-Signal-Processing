import pandas as pd
import numpy as np
from scipy import signal


def find_peaks(X_patient, height=0.5, distance=5, prominence=0.5):
    peaks_list = []
    for i in range(X_patient.shape[1]):
        x = X_patient[:,i]
        peaks_list.append(signal.find_peaks(x, height=height, distance=distance, prominence=prominence)[0])
    return peaks_list


def find_heartbeat_peaks(peaks_arr: list):
    heartbeat_peaks = []
    heartbeats = []
    
    for i in range(len(peaks_arr)):
        peaks_list = peaks_arr[i]
        if len(peaks_list) > 0:
            diff = peaks_list[1:] - peaks_list[:-1]
            median = np.median(diff)
            noisy_ids = []
            
            # Filter for peaks of low distances
            for j in range(diff.shape[0]):
                if median - diff[j] > 15:
                    noisy_ids.append(j+1)
            peaks_list = np.delete(peaks_list, noisy_ids)
            
            # Make sure at least peaks are available per lead to avoid outliers
            if peaks_list.shape[0] < 5:
                peaks_list = np.array([], dtype=int)
            else:
                for j, p in enumerate(peaks_list):
                    new_heartbeat = True
                    if np.any(np.abs(p-heartbeats) < 20):
                        new_heartbeat = False
                    if new_heartbeat:
                        heartbeats.append(p)
        heartbeat_peaks.append(peaks_list)

    # Make sure each heartbeat is supported by at least 3 leads, filter heartbeats and heartbeat_peaks
    safe_heartbeats = []
    for i, h in enumerate(heartbeats):
        confirmed_by_leads = 0
        for j, p in enumerate(heartbeat_peaks):
            if np.any(np.abs(h - p) < 3):
                confirmed_by_leads += 1
        if confirmed_by_leads >= 3:
            safe_heartbeats.append(h)
            
    safe_heartbeats_peaks = []
    for i, hp_i in enumerate(heartbeat_peaks):
        safe_heartbeats_peaks.append([])
        for j, hp_j in enumerate(hp_i):
            if np.any(np.abs(hp_j - p) < 3):
                safe_heartbeats_peaks[i].append(hp_j)

    return safe_heartbeats, safe_heartbeats_peaks


def find_plateau(diff, start, tolerance=1, min_slope=1.0, orientation='left'):
    """
    Find a plateau in the signal. The plateau is reached when the required slope of 0.1 is not fulfilled anymore.
    """
    tol_counter = 0
    valley_index = start
    if orientation == 'left':
        for i in range(start):
            if diff[start-i] > -min_slope:
                tol_counter = tol_counter+1
                valley_index = start-i #+tol_counter
            else:
                tol_counter = 0    
            
            if tol_counter > tolerance:
                break
    elif orientation == 'right':
        for i in range(len(diff) - start - 1):
            if diff[start+i] < min_slope:
                tol_counter = tol_counter+1
                valley_index = start+i #-tol_counter
            else:
                tol_counter = 0
                
            if tol_counter > tolerance:
                break
    else:
        print('Choose orientation between "left" and "right".')
        
    return valley_index



def find_valley(diff, start, tolerance=1, min_dist=3, orientation='left'):
    """
    Find a valley in the signal. The valley is reached when the signal is not falling and the distance min_dist is reached.
    """
    tol_counter = 0
    valley_index = start
    if orientation == 'left':
        for i in range(start):
            if diff[start-i] <= 0 and i > min_dist:
                tol_counter = tol_counter+1
                valley_index = start-i #+tol_counter
            else:
                tol_counter = 0
            if tol_counter > tolerance:
                break
    elif orientation == 'right':
        for i in range(len(diff) - start - 1):
           
            if diff[start+i] >= 0 and i > min_dist:
                tol_counter = tol_counter+1
                valley_index = start+i #-tol_counter
            else:
                tol_counter = 0
            if tol_counter > tolerance:
                break
    else:
        print('Choose orientation between "left" and "right".')
        
    return valley_index



def find_qrs_in_signal(X_patient_lead, peak, tolerance=1):
    limit = 10
    
    if peak - limit < 0:
        lim_left = 0
        peak_relative_id = peak
    else:
        lim_left = peak-limit
        peak_relative_id = limit
    
    if peak + limit < X_patient_lead.shape[0]:
        lim_right = peak + limit
    else:
        lim_right = X_patient_lead.shape[0]
    #print('lim_left ' + str(lim_left) + ' q_start ' + str(q_start + lim_left)
    diff = X_patient_lead[lim_left:lim_right-1] - X_patient_lead[lim_left+1:lim_right]
    q_peak = find_valley(diff, peak_relative_id, orientation='left')
    q_start = find_plateau(diff, q_peak+1, orientation='left')
    

    s_peak = find_valley(diff, peak_relative_id, orientation='right')
    s_end = find_plateau(diff, s_peak+1, orientation='right')
        
    return q_start + lim_left, s_end + lim_left


def find_qrs_complex(X_patient, heartbeat, heartbeat_peaks):
    qrs_start = []
    qrs_end = []
    for i, hb in enumerate(heartbeat):
        qrs_start_i = []
        qrs_end_i = []
        for j, hbp_j in enumerate(heartbeat_peaks):
            for k, hbp_k in enumerate(hbp_j):
                if np.abs(hb-hbp_k) < 3:
                    qrs_start_hbp, qrs_end_hbp = find_qrs_in_signal(X_patient[:,k], peak=hbp_k)
                    qrs_start_i.append(qrs_start_hbp)
                    qrs_end_i.append(qrs_end_hbp)
        qrs_start.append(np.median(qrs_start_i))
        qrs_end.append(np.median(qrs_end_i))
        
    return qrs_start, qrs_end



