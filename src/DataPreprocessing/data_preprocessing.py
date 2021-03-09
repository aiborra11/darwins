import talib
import pandas as pd
import numpy as np
import logging

from datetime import datetime
from typing import (Union)
from sklearn.preprocessing import StandardScaler

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
            self._features.loc[(self._features['hour'] >= 1) & (self._features['hour'] <= 7), 'session'] = 1
            # London
            self._features.loc[(self._features['hour'] >= 8) & (self._features['hour'] <= 15), 'session'] = 2
            # NY + London
            self._features.loc[(self._features['hour'] >= 15) & (self._features['hour'] <= 17), 'session'] = 3
            # NY
            self._features.loc[(self._features['hour'] >= 17) & (self._features['hour'] <= 22), 'session'] = 4
            # Others
            self._features.loc[(self._features['hour'] > 22) | (self._features['hour'] == 0), 'session'] = 5
            logging.info('Adding MARKET SESSION features...')

            return self._features

        except KeyError:
            logging.warning('Something went wrong identifying the market session.')
            print(self._features.columns)
            return self._features

    def technical_indicators(self):
        self._features[['close', 'open', 'max', 'min']] = self._features[['close', 'open', 'max', 'min']].ffill()

        self._features['EMA_3'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=3)
        self._features['EMA_6'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=6)
        self._features['EMA_9'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=9)
        self._features['EMA_20'] = talib.EMA(self._features['close'].to_numpy(), timeperiod=20)
        self._features['SMA_3'] = talib.SMA(self._features['close'].to_numpy(), timeperiod=3)

        self._features = self._features.bfill()
        logging.info('Adding TECHNICAL INDICATORS features...')
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
        self._not_standardizer_columns = config.NOT_STANDARDIZATION_COLS

        self._train_size: float = config.TRAIN_SIZE

        self.standardized_df: Union[pd.DataFrame, None] = None


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
        logging.info(f'Resampling your data to {frequency} candle timeframe')

        return self._resampled_df

    def _get_labels(self):
        if self._candles:
            self._technical_indicators_df['Label'] = np.where((self._technical_indicators_df['close'] <
                                                               self._technical_indicators_df['close'].shift(-1)), 1, 0)
            logging.info('Adding labels to your data...')
        else:
            logging.info('There are no closing values, so we cannot perform any labeling.')

        return self._technical_indicators_df

    def train_test_split(self):
        training_set = self._labeled_data.iloc[:int(len(self._labeled_data) * self._train_size)]
        testing_set = self._labeled_data.iloc[int(len(self._labeled_data) * self._train_size):]
        logging.info('Splitting data in training and testing sets.')
        self._release_memory()
        logging.info('Releasing memory...')

        return training_set, testing_set

    def logarithmic_standardizer(self, df):
        self.standardized_df = df.set_index('timestamp')
        for col in self.standardized_df.columns:
            if col not in self._not_standardizer_columns:
                self.standardized_df.loc[:, col] = np.log(self.standardized_df[col])
        logging.info('Standardizing your data by using a LOGARITHMIC approach...')

        return self.standardized_df

    def scaler_standardizer(self, df, train_df: bool):
        standardized_df = df.copy()
        scaler = StandardScaler()
        for col in standardized_df.columns:
            if col not in self._not_standardizer_columns:
                labeled_data_array_train = np.array(standardized_df[col].ravel()).reshape(-1, 1)
                labeled_data_array_test = np.array(standardized_df[col].ravel()).reshape(-1, 1)
                if train_df:
                    standardized_df.loc[:, col] = scaler.fit_transform(labeled_data_array_train)
                else:
                    standardized_df.loc[:, col] = scaler.fit_transform(labeled_data_array_train)
                    standardized_df.loc[:, col] = scaler.transform(labeled_data_array_test)
        logging.info(f'Standardizing your {"TRAIN" if train_df else "TEST"} data by using a SCALER approach...')

        return standardized_df

    def _release_memory(self):
        self.df = None
        self._resampled_candles = None
        self.resampled_scores = None
        self._resampled_df = None
        self._labeled_data = None
        self._features = None
