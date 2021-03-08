import talib
import pandas as pd
import numpy as np

from datetime import datetime
from typing import (Union)
from sklearn.preprocessing import StandardScaler

from config import config


class FeaturesCalculator:
    def __init__(self):
        self._resampled_df: Union[pd.DataFrame, None] = None
        self._features: Union[pd.DataFrame, None] = None

    def session_identifier(self):
        self._resampled_df = self._resampled_df.reset_index()
        timestamp = self._resampled_df['timestamp'].astype(str).str.split(' ', n=1, expand=True)

        try:
            self._features = self._resampled_df.copy()
            self._features['date'], self._features['hour'] = timestamp[0], timestamp[1]
            self._features['hour'] = (self._features['hour'].str.split(':', 1).str[0]).astype(int)

            # Tokyo
            self._features.loc[(self._resampled_df['hour'] >= 1) & (self._resampled_df['hour'] <= 7), 'session'] = 1
            # London
            self._features.loc[(self._resampled_df['hour'] >= 8) & (self._resampled_df['hour'] <= 15), 'session'] = 2
            # NY + London
            self._features.loc[(self._resampled_df['hour'] >= 15) & (self._resampled_df['hour'] <= 17), 'session'] = 3
            # NY
            self._features.loc[(self._resampled_df['hour'] >= 17) & (self._resampled_df['hour'] <= 22), 'session'] = 4
            # Others
            self._features.loc[(self._resampled_df['hour'] > 22) | (self._resampled_df['hour'] == 0), 'session'] = 5
            return self._features

        except KeyError:
            return self._features

    def technical_indicators(self):
        self._features[['close', 'open', 'max', 'min']] = self._features[['close', 'open', 'max', 'min']].ffill()

        self._features['EMA_3'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=3)
        self._features['EMA_6'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=6)
        self._features['EMA_9'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=9)
        self._features['EMA_20'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=20)
        self._features['SMA_3'] = talib.SMA(self._features['close'].to_numpy(), timeperiod=3)

        self._features = self._features.bfill()
        return self._features


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

        self._resampled_df: Union[pd.DataFrame, None] = self._timeframe_resampler()
        self._features: Union[pd.DataFrame, None] = self.session_identifier() if config.OPERATING_MARKETS \
            else self._resampled_df
        self._technical_indicators_df: pd.DataFrame = self.technical_indicators() if config.TECHNICAL_INDICATORS \
            else self._features

        self._labeled_data: pd.DataFrame = self._get_labels() if self._candles else self._technical_indicators_df
        self._not_standardizer_columns = config.NOT_STANDARDIZARTION_COLS
        # self._log_candles: bool = config.LOG_CANDLES
        # self._log_technical_indicators: bool = config.LOG_TECHNICAL_INDICATORS
        # self._log_df: Union[pd.DataFrame, None] = self.logarithmic_standardizer() if self._candles else \
        #     self._labeled_data.ffill(inplace=True)

        self._train_size: float = config.TRAIN_SIZE

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
            self._resampled_df = pd.merge(self._resampled_candles, self._resampled_scores, on='timestamp', how='left')
        else:
            if self._resampled_candles is not None:
                self._resampled_df = self._resampled_candles
            if self._resampled_scores is not None:
                self._resampled_df = self._resampled_scores

        return self._resampled_df

    def _get_labels(self):
        if self._candles:
            self._technical_indicators_df['Label'] = np.where((self._technical_indicators_df['close'] <
                                                               self._technical_indicators_df['close'].shift(-1)), 1, 0)
        else:
            print('There are no closing values, so we cannot perform any labeling.')
        return self._technical_indicators_df

    # def logarithmic_standardizer(self):
    #     if self._log_candles:
    #         print('Standardizing your data by using a logarithmic approach...')
    #         self._labeled_data[['close', 'max', 'min', 'open']] =\
    #             np.log(self._labeled_data[['close', 'max', 'min', 'open']])
    #
    #         if self._log_technical_indicators:
    #             self.technical_indicators_df[list(self.technical_indicators_df.columns)] = \
    #                 np.log(self.technical_indicators_df[list(self.technical_indicators_df.columns)])
    #
    #         elif self.technical_indicators_df:
    #             self.technical_indicators_df[list(self.technical_indicators_df.columns)] = \
    #                 self.technical_indicators_df[list(self.technical_indicators_df.columns)]
    #         else:
    #             print('Ignoring technical indicators...')
    #     else:
    #         print('Ignoring logarithmic standardizer...')
    #
    #     self._labeled_data = pd.concat([self._labeled_data, self.technical_indicators_df], axis=1)
    #     return self._labeled_data.ffill(inplace=True)

    def logarithmic_standardizer(self):
        print('Standardizing your data by using a LOGARITHMIC approach...')
        self._labeled_data = self._labeled_data.set_index('timestamp')
        for col in self._labeled_data.columns:
            if col not in self._not_standardizer_columns:
                self._labeled_data[col] = np.log(self._labeled_data[col])

        return self._labeled_data.ffill(inplace=True)


    def scaler_standaradizer(self):
        print('Standardizing your data by using a SCALER approach...')
        self._labeled_data = self._labeled_data.set_index('timestamp')
        scaler = StandardScaler()
        for col in self._labeled_data.columns:
            if col not in self._not_standardizer_columns:
                labeled_data_array = np.array(self._labeled_data[col].ravel()).reshape(-1, 1)
                self._labeled_data[col] = scaler.fit_transform(labeled_data_array)

        return self._labeled_data


    def train_test_split(self):
        training_set = self._labeled_data.iloc[:int(len(self._labeled_data) * self._train_size)]
        testing_set = self._labeled_data.iloc[int(len(self._labeled_data) * self._train_size):]
        self._release_memory()
        return training_set, testing_set

    def _release_memory(self):
        self._resampled_candles = None
        self.resampled_scores = None
        self._resampled_df = None
        self._labeled_data = None
        self._features = None
