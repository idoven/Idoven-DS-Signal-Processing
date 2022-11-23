#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Carla Guillén Pingarrón"

import ast
import collections
import pathlib
from enum import Enum

import pandas as pd
import wfdb

SamplingRate = collections.namedtuple('SamplingRate', 'filename rate')


class SamplingRates(Enum):
    LOW = SamplingRate("filename_lr", 100)  # 100 Hz
    HIGH = SamplingRate("filename_hr", 500)  # 500 Hz


DEFAULT_PATH = pathlib.Path(__file__).parent.resolve() / 'ptb-xl' / '1.0.2'
DEFAULT_SAMPLING_RATE = SamplingRates.LOW


class Dataset():
    """
    Class for accessing the ptb-xl dataset easily
    """

    path: pathlib.Path
    """path to the ptb-xl dataset"""

    sampling_rate: SamplingRates
    """sampling rate for waveform data"""

    db: pd.DataFrame
    """
    dataframe with ptb-xl metadata together with the wfdb records
    (signal + descriptors)
    """

    def __init__(
        self,
        path: pathlib.Path = DEFAULT_PATH,
        sampling_rate: SamplingRates = DEFAULT_SAMPLING_RATE
    ) -> None:
        self.path = path
        self.sampling_rate = sampling_rate
        self.annotations = self._load_annotations()
        self.db = self._load_db()

    def _load_annotations(self):
        scp_statements = self.path / 'scp_statements.csv'
        annotations = pd.read_csv(str(scp_statements), index_col=0)
        return annotations

    def _load_db(self):
        # Load database. ecg_id is unique, so we use it as index.
        ptbxl_database = self.path / 'ptbxl_database.csv'
        db = pd.read_csv(str(ptbxl_database), index_col='ecg_id')
        # scp_codes is a dictionary so we evaluate it as python code and
        # transform it from str to dict
        db['scp_codes'] = db['scp_codes'].apply(ast.literal_eval)
        # Load wfdb records for each ecg_id into ptbxl_df
        db['record'] = db[self.sampling_rate.value.filename].apply(
            lambda filename: wfdb.rdrecord(str(self.path / filename)))

        return db
