import wfdb
import ast
import numpy as np
import pandas as pd


def load_raw_data(df:pd.DataFrame, sampling_rate: int, path: str):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def load_signal_and_annotations(sampling_rate: int, path: str):
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    return X, Y


def load_diagnostic_aggregation(path: str, Y: pd.DataFrame):
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    return Y

def load_test_fold(X: pd.DataFrame, Y: pd.DataFrame, test_fold: int):
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    Y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    Y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return X_train, X_test, Y_train, Y_test