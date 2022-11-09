import pandas as pd
import numpy as np
import ast
import wfdb
from typing import List, Dict

def load_diagnostic(path:str)->pd.DataFrame:
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    return agg_df

def aggregate_diagnostic(y_dic:Dict, agg_df:pd.DataFrame)->List:
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def load_annotation(agg_df:pd.DataFrame, path:str)->pd.DataFrame:
    df_db = pd.read_csv(path+'ptbxl_database.csv')
    df_db.scp_codes = df_db.scp_codes.apply(lambda x: ast.literal_eval(x))
    df_db = df_db.assign(diagnostic_superclass=\
                         df_db.scp_codes.apply(lambda x: aggregate_diagnostic(x,agg_df)))                     
    return df_db

def load_ecg_data(df:pd.DataFrame, sampling_rate:int, path:str)->pd.DataFrame:
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def clean_df_annotation(df:pd.DataFrame)->pd.DataFrame:
    df.drop("ecg_id", axis=1, inplace=True)
    df = df.assign(ecg_id = df.index)
    df.r_peaks = df.r_peaks\
                            .apply(lambda col: list(map(int,col.strip('[]').split())))
    df = df.assign(len_diagn = df.diagnostic_superclass.map(lambda x:len(x)))
    df = df[df.len_diagn!=0]
    df = df.explode(['diagnostic_superclass'])
    df = df.assign(sex_cat = df.sex.map({1: 'Female', 0: 'Male'}))
    df = df.assign(diagnostic_binary_superclass=df.diagnostic_superclass\
                         .apply(lambda x: 'ANOMALY' if x in ['MI', 'STTC', 'HYP', 'CD'] else 'NORMAL'))
    df.drop("len_diagn", axis=1,inplace=True)
    df = df[df.age<=100]
    
    return df