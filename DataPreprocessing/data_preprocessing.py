import talib
import pandas as pd
import numpy as np

from datetime import datetime
from typing import (Union)

from Configuration.config import config
from DataLoader.data_loader import DataLoader


class DataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self._df: [pd.DataFrame, None] = df

        self.timeframe: str = config.TIMEFRAME
        self._df: pd.DataFrame = self._datetime_indexer()
        self._candles: bool = config.CANDLES_DATA
        self._scores: bool = config.SCORES_DATA
        self._resampled_candles: Union[pd.DataFrame, None] = None
        self._resampled_scores: Union[pd.DataFrame, None] = None
        self._resampled_df: Union[pd.DataFrame, None] = self._timeframe_resampler()
        self._labeled_data: pd.DataFrame = self._get_labels()

        self._log_cols: Union[list, None] = config.LOG_COLS
        self._log_df = self._labeled_data if not self._log_cols else self._get_logarithmic_data()
        self.train_size: float = config.TRAIN_SIZE

    def _datetime_indexer(self):
        self._df = self._df.reset_index().rename(columns={'index': 'timestamp'})
        self._df['timestamp'] = self._df['timestamp'].map(lambda t: datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S'))
        self._df = self._df.set_index('timestamp')
        return self._df

    def _timeframe_resampler(self):
        frequency = self.timeframe
        if self._candles:
            self._resampled_candles = (self._df.resample(frequency).agg({'open': 'first', 'max':
                                                                       'max', 'min': 'min', 'close': 'last'}))
        if self._scores:
            self._resampled_scores = self._df[self._df.columns[~self._df.columns.isin(['open', 'close', 'max', 'min'])]]\
                                                                                            .resample(frequency).mean()
        if self._resampled_candles is not None and self._resampled_scores is not None:
            self._resampled_df = pd.merge(self._resampled_candles, self._resampled_scores, on='timestamp', how='left')
        else:
            if self._resampled_candles is not None:
                self._resampled_df = self._resampled_candles
            if self._resampled_scores is not None:
                self._resampled_df = self._resampled_scores
        return self._resampled_df

    def _get_labels(self):
        if self._candles:
            self._resampled_df['Label'] = np.where((self._resampled_df['close'] <
                                                    self._resampled_df['close'].shift(-1)), 1, 0)
        else:
            print('There are no closing values, so we cannot perform any labeling.')
        return self._resampled_df

    def _get_logarithmic_data(self):
        self._labeled_data[config.LOG_COLS] = np.log(self._labeled_data[self._log_cols])
        return self._labeled_data

    def train_test_split(self):
        training_set = self._log_df.iloc[:int(len(self._log_df)*self.train_size)]
        testing_set = self._log_df.iloc[int(len(self._log_df)*self.train_size):]
        self._release_memory()
        return training_set, testing_set

    def _release_memory(self):
        self._df = None
        self._resampled_candles = None
        self.resampled_scores = None
        self._resampled_df = None
        self._labeled_data = None
        self._log_df = None


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


