# -*- coding: utf-8 -*-

import numpy
import pandas
import os
import wfdb
import tqdm.notebook


def aggregate_heart_rate(peaks: numpy.array):
  rate = len(peaks) * 6 # due to 10 seconds length

  return rate


def aggregate_RR_interval_variability(peaks: numpy.array):
  intervals = numpy.diff(peaks)
  variability = numpy.std(intervals) / numpy.mean(intervals) # coefficient of variation

  return variability


def aggregate_superclasses(data: pandas.Series, data_scp: pandas.DataFrame):
  superclasses = []

  for key in data.keys():
    if key in data_scp.index:
      superclass = data_scp.loc[key].diagnostic_class
      superclasses.append(superclass)

  return pandas.Series([list(set(superclasses))])


def load_ECGs(mode: str, superclass: str, data: pandas.DataFrame, n: int):
  file = f'records100/{mode}/{superclass}.npy'

  if os.path.isfile(file):

    with open(file, 'rb') as f:
      ECG = numpy.load(f)

  else:

    if mode == "training":
      data = data[~data.strat_fold.isin([9, 10])]
    elif mode == "validation":
      data = data[data.strat_fold == 9]
    else:
      data = data[data.strat_fold == 10]

    data = data[data.binary_diagnostic_superclass == superclass][:n]
    ECG = numpy.array([wfdb.rdsamp('physionet.org/files/ptb-xl/1.0.2/' + f_lr)[0] for f_lr in tqdm.notebook.tqdm(data.filename_lr[:n], desc="Extracting ECGs")])
    os.makedirs('records100', exist_ok = True)
    os.makedirs(f'records100/{mode}', exist_ok = True)

    with open(f'records100/{mode}/{superclass}.npy', 'wb') as f:
      numpy.save(f, ECG)

  return ECG