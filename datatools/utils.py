import wfdb
import numpy as np
import pandas as pd
import ast

def load_raw_data(df, sampling_rate, path):
    """ Method from the physionet dataset (example_physionet.py)"""
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def get_annotations(path: str, num_epochs = 10):
    """ Method to load the annotations from the csv
    Args:
    - path: path where the annotations file is located
    - num_epochs: number of rows to include, so not all the 
    dataset has to be loaded each time
    """
    
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y = Y.iloc[0:num_epochs,:]
    
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    return Y

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
