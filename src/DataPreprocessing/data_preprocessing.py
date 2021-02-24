import talib
import pandas as pd
import numpy as np

from datetime import datetime
from typing import (Union)

from Configuration.config import config

class FeaturesCalculator:
    def __init__(self):
        # self.df = self.session_identifier(df) if config.OPERATING_MARKETS else df
        # self.df = self.technical_indicators() if config.TECHNICAL_INDICATORS else df
        self.resampled_df: Union[pd.DataFrame, None] = None

    def session_identifier(self):
        self.resampled_df = self.resampled_df.reset_index()
        self.resampled_df['date'], self.resampled_df['hour'] = self.resampled_df['timestamp'].astype(str)\
                                                                                             .str.split(' ', 1).str
        self.resampled_df['hour'] = self.resampled_df['hour'].str.split(':', 1).str[0]
        self.resampled_df['hour'] = self.resampled_df['hour'].astype(int)

        self.resampled_df.loc[(self.resampled_df['hour'] >= 1) & (self.resampled_df['hour'] <= 7), 'session'] = 1             # Tokyo
        self.resampled_df.loc[(self.resampled_df['hour'] >= 8) & (self.resampled_df['hour'] <= 15), 'session'] = 2            # London
        self.resampled_df.loc[(self.resampled_df['hour'] >= 15) & (self.resampled_df['hour'] <= 17), 'session'] = 3           # NY + London
        self.resampled_df.loc[(self.resampled_df['hour'] >= 17) & (self.resampled_df['hour'] <= 22), 'session'] = 4           # NY
        self.resampled_df.loc[(self.resampled_df['hour'] > 22) | (self.resampled_df['hour'] == 0), 'session'] = 5             # Others

        return self.resampled_df

    def technical_indicators(self):
        self.resampled_df['EMA_3'] = talib.EMA(self.resampled_df['close'].values, timeperiod=3)
        self.resampled_df['EMA_6'] = talib.EMA(self.resampled_df['close'].values, timeperiod=6)
        self.resampled_df['EMA_9'] = talib.EMA(self.resampled_df['close'].values, timeperiod=9)
        self.resampled_df['EMA_20'] = talib.EMA(self.resampled_df['close'].values, timeperiod=20)

        return self.resampled_df


class DataPreprocessing(FeaturesCalculator):
    def __init__(self, data):
        super().__init__()
        self.df: [pd.DataFrame, None] = data

        self.timeframe: str = config.TIMEFRAME
        self.df: pd.DataFrame = self._datetime_indexer()

        self._candles: bool = config.CANDLES_DATA
        self._scores: bool = config.SCORES_DATA
        self._resampled_candles: Union[pd.DataFrame, None] = None
        self._resampled_scores: Union[pd.DataFrame, None] = None
        self.resampled_df: Union[pd.DataFrame, None] = self._timeframe_resampler()

        self.resampled_df = self.session_identifier() if config.OPERATING_MARKETS else self.resampled_df
        print(self.resampled_df)

        self.resampled_df = self.technical_indicators() if config.TECHNICAL_INDICATORS else self.resampled_df
        print(self.resampled_df)

        self._labeled_data: pd.DataFrame = self._get_labels()

        self._log_cols: Union[list, None] = config.LOG_COLS
        self.log_df = self._labeled_data if not self._log_cols else self._get_logarithmic_data()
        self.train_size: float = config.TRAIN_SIZE

    def _datetime_indexer(self):
        self.df['index'] = self.df['index'].map(lambda t: datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S'))
        self.df = self.df.rename(columns={'index': 'timestamp'}).set_index('timestamp')
        return self.df

    def _timeframe_resampler(self):
        frequency = self.timeframe
        if self._candles:
            self._resampled_candles = (self.df.resample(frequency).agg({'open': 'first', 'max':
                                                                        'max', 'min': 'min', 'close': 'last'}))
        if self._scores:
            self._resampled_scores = self.df[self.df.columns[~self.df.columns.isin(['open', 'close', 'max', 'min'])]]\
                                                                                            .resample(frequency).mean()
        if self._resampled_candles is not None and self._resampled_scores is not None:
            self.resampled_df = pd.merge(self._resampled_candles, self._resampled_scores, on='timestamp', how='left')
        else:
            if self._resampled_candles is not None:
                self.resampled_df = self._resampled_candles
            if self._resampled_scores is not None:
                self.resampled_df = self._resampled_scores

        return self.resampled_df

    def _get_labels(self):
        if self._candles:
            self.resampled_df['Label'] = np.where((self.resampled_df['close'] <
                                                   self.resampled_df['close'].shift(-1)), 1, 0)
        else:
            print('There are no closing values, so we cannot perform any labeling.')
        return self.resampled_df

    def _get_logarithmic_data(self):
        self._labeled_data[config.LOG_COLS] = np.log(self._labeled_data[self._log_cols])
        return self._labeled_data

    def train_test_split(self, df):
        training_set = self.df.iloc[:int(len(self.df) * self.train_size)]
        testing_set = self.df.iloc[int(len(self.df) * self.train_size):]
        self._release_memory()
        return training_set, testing_set

    def _release_memory(self):
        self._df = None
        self._resampled_candles = None
        self.resampled_scores = None
        self.resampled_df = None
        self._labeled_data = None
        self.log_df = None

