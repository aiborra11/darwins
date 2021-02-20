import talib
import pandas as pd
import numpy as np

from typing import (Union)

from Configuration.config import Config



class DataPreprocessing:
    def __init__(self, df: pd.DataFrame, timeframe: str):
        config = Config()
        self.df = df
        self.timeframe = timeframe
        self.train_size: float = config.TRAIN_SIZE
        self.log_cols: Union[list, None] = config.LOG_COLS

    def get_labels(self):
        self.df['Label'] = np.where((self.df['close'] < self.df['close'].shift(-1)), 1, 0)
        return self.df

    def get_logarithmic_data(self):
        self.df = np.log(self.df[self.log_cols])
        return self.df

    def train_test_split(self, df: pd.DataFrame):
        data_train = self.df .iloc[:int(len(self.df)*self.train_size)]
        data_test = self.df .iloc[int(len(self.df)*self.train_size):]

        return data_train, data_test


class FeaturesCalculator(DataPreprocessing):
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame() = df

    def session_identifier(self):

        self.df['hour'] = self.df['hour'].str.split(':', 1).str[0]
        self.df['hour'] = self.df['hour'].astype(int)

        self.df.loc[(self.df['hour'] >= 1) & (self.df['hour'] <= 7), 'session'] = 1             # Tokyo
        self.df.loc[(self.df['hour'] >= 8) & (self.df['hour'] <= 15), 'session'] = 2            # London
        self.df.loc[(self.df['hour'] >= 15) & (self.df['hour'] <= 17), 'session'] = 3           # NY + London
        self.df.loc[(self.df['hour'] >= 17) & (self.df['hour'] <= 22), 'session'] = 4           # NY
        self.df.loc[(self.df['hour'] > 22) | (self.df['hour'] == 0), 'session'] = 5             # Others

        return self.df

    def technical_indicators(self):

        self.df['EMA_3'] = talib.EMA(self.df['close'].values, timeperiod=3)
        self.df['EMA_6'] = talib.EMA(self.df['close'].values, timeperiod=6)
        self.df['EMA_9'] = talib.EMA(self.df['close'].values, timeperiod=9)
        self.df['EMA_20'] = talib.EMA(self.df['close'].values, timeperiod=20)

        return self.df


